/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "barnes_hut_kernels.cuh"
#include "utils.cuh"
#include <cuml/common/logger.hpp>
#include <cuml/manifold/tsne.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/eltwise.hpp>

namespace ML {
namespace TSNE {

/**
 * @brief Fast Dimensionality reduction via TSNE using the Barnes Hut O(NlogN) approximation.
 * @param[in] VAL: The values in the attractive forces COO matrix.
 * @param[in] COL: The column indices in the attractive forces COO matrix.
 * @param[in] ROW: The row indices in the attractive forces COO matrix.
 * @param[in] NNZ: The number of non zeros in the attractive forces COO matrix.
 * @param[in] handle: The GPU handle.
 * @param[out] Y: The final embedding. Will overwrite this internally.
 * @param[in] n: Number of rows in data X.
 * @param[in] params: Parameters for TSNE model.
 */

template <typename value_idx, typename value_t>
value_t Barnes_Hut(value_t* VAL,
                   const value_idx* COL,
                   const value_idx* ROW,
                   const value_idx NNZ,
                   const raft::handle_t& handle,
                   value_t* Y,
                   const value_idx n,
                   const TSNEParams& params)
{
  cudaStream_t stream = handle.get_stream();

  value_t kl_div = 0;

  // Get device properites
  //---------------------------------------------------
  const int blocks = raft::getMultiProcessorCount();

  auto nnodes = n * 2;
  if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
  while ((nnodes & (32 - 1)) != 0)
    nnodes++;
  nnodes--;
  CUML_LOG_DEBUG("N_nodes = %d blocks = %d", nnodes, blocks);

  // Allocate more space
  // rmm::device_uvector<unsigned> errl(1, stream);
  rmm::device_scalar<unsigned> limiter(stream);
  rmm::device_scalar<value_idx> maxdepthd(stream);
  rmm::device_scalar<value_idx> bottomd(stream);
  rmm::device_scalar<value_t> radiusd(stream);

  BH::InitializationKernel<<<1, 1, 0, stream>>>(/*errl.data(),*/
                                                limiter.data(),
                                                maxdepthd.data(),
                                                radiusd.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  const value_idx FOUR_NNODES = 4 * nnodes;
  const value_idx FOUR_N      = 4 * n;
  const float theta_squared   = params.theta * params.theta;
  const value_idx NNODES      = nnodes;

  // Actual allocations
  rmm::device_uvector<value_idx> startl(nnodes + 1, stream);
  rmm::device_uvector<value_idx> childl((nnodes + 1) * 4, stream);
  rmm::device_uvector<value_t> massl(nnodes + 1, stream);

  thrust::device_ptr<value_t> begin_massl = thrust::device_pointer_cast(massl.data());
  thrust::fill(thrust::cuda::par.on(stream), begin_massl, begin_massl + (nnodes + 1), 1.0f);

  rmm::device_uvector<value_t> maxxl(blocks * FACTOR1, stream);
  rmm::device_uvector<value_t> maxyl(blocks * FACTOR1, stream);
  rmm::device_uvector<value_t> minxl(blocks * FACTOR1, stream);
  rmm::device_uvector<value_t> minyl(blocks * FACTOR1, stream);

  // SummarizationKernel
  rmm::device_uvector<value_idx> countl(nnodes + 1, stream);

  // SortKernel
  rmm::device_uvector<value_idx> sortl(nnodes + 1, stream);

  // RepulsionKernel
  rmm::device_uvector<value_t> rep_forces((nnodes + 1) * 2, stream);
  rmm::device_uvector<value_t> attr_forces(n * 2, stream);  // n*2 double for reduction sum

  rmm::device_scalar<value_t> Z_norm(stream);

  rmm::device_scalar<value_t> radiusd_squared(stream);

  // Apply
  rmm::device_uvector<value_t> gains_bh(n * 2, stream);

  thrust::device_ptr<value_t> begin_gains_bh = thrust::device_pointer_cast(gains_bh.data());
  thrust::fill(handle.get_thrust_policy(), begin_gains_bh, begin_gains_bh + (n * 2), 1.0f);

  rmm::device_uvector<value_t> old_forces(n * 2, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(old_forces.data(), 0, sizeof(value_t) * n * 2, stream));

  rmm::device_uvector<value_t> YY((nnodes + 1) * 2, stream);
  if (params.initialize_embeddings) {
    random_vector(YY.data(), -0.0001f, 0.0001f, (nnodes + 1) * 2, stream, params.random_state);
  } else {
    raft::copy(YY.data(), Y, n, stream);
    raft::copy(YY.data() + nnodes + 1, Y + n, n, stream);
  }

  rmm::device_uvector<value_t> tmp(NNZ, stream);
  value_t* Qs      = tmp.data();
  value_t* KL_divs = tmp.data();

  // Set cache levels for faster algorithm execution
  //---------------------------------------------------
  RAFT_CUDA_TRY(
    cudaFuncSetCacheConfig(BH::BoundingBoxKernel<value_idx, value_t>, cudaFuncCachePreferShared));
  RAFT_CUDA_TRY(
    cudaFuncSetCacheConfig(BH::TreeBuildingKernel<value_idx, value_t>, cudaFuncCachePreferL1));
  RAFT_CUDA_TRY(cudaFuncSetCacheConfig(BH::ClearKernel1<value_idx>, cudaFuncCachePreferL1));
  RAFT_CUDA_TRY(
    cudaFuncSetCacheConfig(BH::ClearKernel2<value_idx, value_t>, cudaFuncCachePreferL1));
  RAFT_CUDA_TRY(
    cudaFuncSetCacheConfig(BH::SummarizationKernel<value_idx, value_t>, cudaFuncCachePreferShared));
  RAFT_CUDA_TRY(cudaFuncSetCacheConfig(BH::SortKernel<value_idx>, cudaFuncCachePreferL1));
  RAFT_CUDA_TRY(
    cudaFuncSetCacheConfig(BH::RepulsionKernel<value_idx, value_t>, cudaFuncCachePreferL1));
  RAFT_CUDA_TRY(
    cudaFuncSetCacheConfig(BH::attractive_kernel_bh<value_idx, value_t>, cudaFuncCachePreferL1));
  RAFT_CUDA_TRY(
    cudaFuncSetCacheConfig(BH::IntegrationKernel<value_idx, value_t>, cudaFuncCachePreferL1));
  // Do gradient updates
  //---------------------------------------------------
  CUML_LOG_DEBUG("Start gradient updates!");

  value_t momentum      = params.pre_momentum;
  value_t learning_rate = params.pre_learning_rate;

  for (int iter = 0; iter < params.max_iter; iter++) {
    RAFT_CUDA_TRY(cudaMemsetAsync(static_cast<void*>(rep_forces.data()),
                                  0,
                                  rep_forces.size() * sizeof(*rep_forces.data()),
                                  stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(static_cast<void*>(attr_forces.data()),
                                  0,
                                  attr_forces.size() * sizeof(*attr_forces.data()),
                                  stream));

    BH::Reset_Normalization<<<1, 1, 0, stream>>>(
      Z_norm.data(), radiusd_squared.data(), bottomd.data(), NNODES, radiusd.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    if (iter == params.exaggeration_iter) {
      momentum = params.post_momentum;

      // Divide perplexities
      const value_t div = 1.0f / params.early_exaggeration;
      raft::linalg::scalarMultiply(VAL, VAL, div, NNZ, stream);

      learning_rate = params.post_learning_rate;
    }

    START_TIMER;
    BH::BoundingBoxKernel<<<blocks * FACTOR1, THREADS1, 0, stream>>>(startl.data(),
                                                                     childl.data(),
                                                                     massl.data(),
                                                                     YY.data(),
                                                                     YY.data() + nnodes + 1,
                                                                     maxxl.data(),
                                                                     maxyl.data(),
                                                                     minxl.data(),
                                                                     minyl.data(),
                                                                     FOUR_NNODES,
                                                                     NNODES,
                                                                     n,
                                                                     limiter.data(),
                                                                     radiusd.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    END_TIMER(BoundingBoxKernel_time);

    START_TIMER;
    BH::ClearKernel1<<<blocks, 1024, 0, stream>>>(childl.data(), FOUR_NNODES, FOUR_N);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    END_TIMER(ClearKernel1_time);

    START_TIMER;
    BH::TreeBuildingKernel<<<blocks * FACTOR2, THREADS2, 0, stream>>>(
      /*errl.data(),*/ childl.data(),
      YY.data(),
      YY.data() + nnodes + 1,
      NNODES,
      n,
      maxdepthd.data(),
      bottomd.data(),
      radiusd.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    END_TIMER(TreeBuildingKernel_time);

    START_TIMER;
    BH::ClearKernel2<<<blocks * 1, 1024, 0, stream>>>(
      startl.data(), massl.data(), NNODES, bottomd.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    END_TIMER(ClearKernel2_time);

    START_TIMER;
    BH::SummarizationKernel<<<blocks * FACTOR3, THREADS3, 0, stream>>>(countl.data(),
                                                                       childl.data(),
                                                                       massl.data(),
                                                                       YY.data(),
                                                                       YY.data() + nnodes + 1,
                                                                       NNODES,
                                                                       n,
                                                                       bottomd.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    END_TIMER(SummarizationKernel_time);

    START_TIMER;
    BH::SortKernel<<<blocks * FACTOR4, THREADS4, 0, stream>>>(
      sortl.data(), countl.data(), startl.data(), childl.data(), NNODES, n, bottomd.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    END_TIMER(SortKernel_time);

    START_TIMER;
    BH::RepulsionKernel<<<blocks * FACTOR5, THREADS5, 0, stream>>>(
      /*errl.data(),*/ params.theta,
      params.epssq,
      sortl.data(),
      childl.data(),
      massl.data(),
      YY.data(),
      YY.data() + nnodes + 1,
      rep_forces.data(),
      rep_forces.data() + nnodes + 1,
      Z_norm.data(),
      theta_squared,
      NNODES,
      FOUR_NNODES,
      n,
      radiusd_squared.data(),
      maxdepthd.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    END_TIMER(RepulsionTime);

    START_TIMER;
    BH::Find_Normalization<<<1, 1, 0, stream>>>(Z_norm.data(), n);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    END_TIMER(Reduction_time);

    START_TIMER;
    // TODO: Calculate Kullback-Leibler divergence
    // For general embedding dimensions
    bool last_iter = iter == params.max_iter - 1;

    BH::attractive_kernel_bh<<<raft::ceildiv(NNZ, (value_idx)1024), 1024, 0, stream>>>(
      VAL,
      COL,
      ROW,
      YY.data(),
      YY.data() + nnodes + 1,
      attr_forces.data(),
      attr_forces.data() + n,
      last_iter ? Qs : nullptr,
      NNZ,
      fmaxf(params.dim - 1, 1));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    END_TIMER(attractive_time);

    if (last_iter) {
      kl_div = compute_kl_div(VAL, Qs, KL_divs, NNZ, stream);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    START_TIMER;
    BH::IntegrationKernel<<<blocks * FACTOR6, THREADS6, 0, stream>>>(learning_rate,
                                                                     momentum,
                                                                     params.early_exaggeration,
                                                                     YY.data(),
                                                                     YY.data() + nnodes + 1,
                                                                     attr_forces.data(),
                                                                     attr_forces.data() + n,
                                                                     rep_forces.data(),
                                                                     rep_forces.data() + nnodes + 1,
                                                                     gains_bh.data(),
                                                                     gains_bh.data() + n,
                                                                     old_forces.data(),
                                                                     old_forces.data() + n,
                                                                     Z_norm.data(),
                                                                     n);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    END_TIMER(IntegrationKernel_time);
  }
  PRINT_TIMES;

  // Copy final YY into true output Y
  raft::copy(Y, YY.data(), n, stream);
  raft::copy(Y + n, YY.data() + nnodes + 1, n, stream);

  return kl_div;
}

}  // namespace TSNE
}  // namespace ML
