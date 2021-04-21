/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <cuml/common/device_buffer.hpp>
#include <cuml/common/logger.hpp>
#include "barnes_hut_kernels.cuh"
#include "utils.cuh"

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
 * @param[in] theta: repulsion threshold
 * @param[in] epssq: A tiny jitter to promote numerical stability.
 * @param[in] early_exaggeration: How much pressure to apply to clusters to spread out during the exaggeration phase.
 * @param[in] exaggeration_iter: How many iterations you want the early pressure to run for. If late exaggeration is used, it will begin after this number of iterations. 
 * @param[in] min_gain: Rounds up small gradient updates.
 * @param[in] pre_learning_rate: The learning rate during the exaggeration phase.
 * @param[in] post_learning_rate: The learning rate after the exaggeration phase.
 * @param[in] max_iter: The maximum number of iterations TSNE should run for.
 * @param[in] min_grad_norm: The smallest gradient norm TSNE should terminate on. 
              This argument is currently ignored.
 * @param[in] pre_momentum: The momentum used during the exaggeration phase.
 * @param[in] post_momentum: The momentum used after the exaggeration phase.
 * @param[in] random_state: Set this to -1 for pure random intializations or >= 0 for reproducible outputs.
 */

template <typename value_idx, typename value_t>
void Barnes_Hut(value_t *VAL, const value_idx *COL, const value_idx *ROW,
                const value_idx NNZ, const raft::handle_t &handle, value_t *Y,
                const value_idx n, const float theta = 0.5f,
                const float epssq = 0.0025,
                const float early_exaggeration = 12.0f,
                const int exaggeration_iter = 250, const float min_gain = 0.01f,
                const float pre_learning_rate = 200.0f,
                const float post_learning_rate = 500.0f,
                const int max_iter = 1000, const float min_grad_norm = 1e-7,
                const float pre_momentum = 0.5, const float post_momentum = 0.8,
                const long long random_state = -1,
                const bool initialize_embeddings = true) {
  auto d_alloc = handle.get_device_allocator();
  cudaStream_t stream = handle.get_stream();

  // Get device properites
  //---------------------------------------------------
  const int blocks = raft::getMultiProcessorCount();

  auto nnodes = n * 2;
  if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
  while ((nnodes & (32 - 1)) != 0) nnodes++;
  nnodes--;
  CUML_LOG_DEBUG("N_nodes = %d blocks = %d", nnodes, blocks);

  // Allocate more space
  // MLCommon::device_buffer<unsigned> errl(d_alloc, stream, 1);
  MLCommon::device_buffer<unsigned> limiter(d_alloc, stream, 1);
  MLCommon::device_buffer<value_idx> maxdepthd(d_alloc, stream, 1);
  MLCommon::device_buffer<value_idx> bottomd(d_alloc, stream, 1);
  MLCommon::device_buffer<value_t> radiusd(d_alloc, stream, 1);

  BH::InitializationKernel<<<1, 1, 0, stream>>>(/*errl.data(),*/
                                                limiter.data(),
                                                maxdepthd.data(),
                                                radiusd.data());
  CUDA_CHECK(cudaPeekAtLastError());

  const value_idx FOUR_NNODES = 4 * nnodes;
  const value_idx FOUR_N = 4 * n;
  const float theta_squared = theta * theta;
  const value_idx NNODES = nnodes;

  // Actual allocations
  MLCommon::device_buffer<value_idx> startl(d_alloc, stream, nnodes + 1);
  MLCommon::device_buffer<value_idx> childl(d_alloc, stream, (nnodes + 1) * 4);
  MLCommon::device_buffer<value_t> massl(d_alloc, stream, nnodes + 1);

  thrust::device_ptr<value_t> begin_massl =
    thrust::device_pointer_cast(massl.data());
  thrust::fill(thrust::cuda::par.on(stream), begin_massl,
               begin_massl + (nnodes + 1), 1.0f);

  MLCommon::device_buffer<value_t> maxxl(d_alloc, stream, blocks * FACTOR1);
  MLCommon::device_buffer<value_t> maxyl(d_alloc, stream, blocks * FACTOR1);
  MLCommon::device_buffer<value_t> minxl(d_alloc, stream, blocks * FACTOR1);
  MLCommon::device_buffer<value_t> minyl(d_alloc, stream, blocks * FACTOR1);

  // SummarizationKernel
  MLCommon::device_buffer<value_idx> countl(d_alloc, stream, nnodes + 1);

  // SortKernel
  MLCommon::device_buffer<value_idx> sortl(d_alloc, stream, nnodes + 1);

  // RepulsionKernel
  MLCommon::device_buffer<value_t> rep_forces(d_alloc, stream,
                                              (nnodes + 1) * 2);
  MLCommon::device_buffer<value_t> attr_forces(
    d_alloc, stream, n * 2);  // n*2 double for reduction sum

  MLCommon::device_buffer<value_t> Z_norm(d_alloc, stream, 1);

  MLCommon::device_buffer<value_t> radiusd_squared(d_alloc, stream, 1);

  // Apply
  MLCommon::device_buffer<value_t> gains_bh(d_alloc, stream, n * 2);

  thrust::device_ptr<value_t> begin_gains_bh =
    thrust::device_pointer_cast(gains_bh.data());
  thrust::fill(thrust::cuda::par.on(stream), begin_gains_bh,
               begin_gains_bh + (n * 2), 1.0f);

  MLCommon::device_buffer<value_t> old_forces(d_alloc, stream, n * 2);
  CUDA_CHECK(
    cudaMemsetAsync(old_forces.data(), 0, sizeof(value_t) * n * 2, stream));

  MLCommon::device_buffer<value_t> YY(d_alloc, stream, (nnodes + 1) * 2);
  if (initialize_embeddings) {
    random_vector(YY.data(), -0.0001f, 0.0001f, (nnodes + 1) * 2, stream,
                  random_state);
  } else {
    raft::copy(YY.data(), Y, n, stream);
    raft::copy(YY.data() + nnodes + 1, Y + n, n, stream);
  }

  // Set cache levels for faster algorithm execution
  //---------------------------------------------------
  CUDA_CHECK(cudaFuncSetCacheConfig(BH::BoundingBoxKernel<value_idx, value_t>,
                                    cudaFuncCachePreferShared));
  CUDA_CHECK(cudaFuncSetCacheConfig(BH::TreeBuildingKernel<value_idx, value_t>,
                                    cudaFuncCachePreferL1));
  CUDA_CHECK(
    cudaFuncSetCacheConfig(BH::ClearKernel1<value_idx>, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(BH::ClearKernel2<value_idx, value_t>,
                                    cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(BH::SummarizationKernel<value_idx, value_t>,
                                    cudaFuncCachePreferShared));
  CUDA_CHECK(
    cudaFuncSetCacheConfig(BH::SortKernel<value_idx>, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(BH::RepulsionKernel<value_idx, value_t>,
                                    cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(
    BH::attractive_kernel_bh<value_idx, value_t>, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(BH::IntegrationKernel<value_idx, value_t>,
                                    cudaFuncCachePreferL1));
  // Do gradient updates
  //---------------------------------------------------
  CUML_LOG_DEBUG("Start gradient updates!");

  value_t momentum = pre_momentum;
  value_t learning_rate = pre_learning_rate;

  for (int iter = 0; iter < max_iter; iter++) {
    CUDA_CHECK(cudaMemsetAsync(static_cast<void *>(rep_forces.data()), 0,
                               rep_forces.size() * sizeof(*rep_forces.data()),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(static_cast<void *>(attr_forces.data()), 0,
                               attr_forces.size() * sizeof(*attr_forces.data()),
                               stream));

    BH::Reset_Normalization<<<1, 1, 0, stream>>>(
      Z_norm.data(), radiusd_squared.data(), bottomd.data(), NNODES,
      radiusd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    if (iter == exaggeration_iter) {
      momentum = post_momentum;

      // Divide perplexities
      const value_t div = 1.0f / early_exaggeration;
      raft::linalg::scalarMultiply(VAL, VAL, div, NNZ, stream);

      learning_rate = post_learning_rate;
    }

    START_TIMER;
    BH::BoundingBoxKernel<<<blocks * FACTOR1, THREADS1, 0, stream>>>(
      startl.data(), childl.data(), massl.data(), YY.data(),
      YY.data() + nnodes + 1, maxxl.data(), maxyl.data(), minxl.data(),
      minyl.data(), FOUR_NNODES, NNODES, n, limiter.data(), radiusd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(BoundingBoxKernel_time);

    START_TIMER;
    BH::ClearKernel1<<<blocks, 1024, 0, stream>>>(childl.data(), FOUR_NNODES,
                                                  FOUR_N);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(ClearKernel1_time);

    START_TIMER;
    BH::TreeBuildingKernel<<<blocks * FACTOR2, THREADS2, 0, stream>>>(
      /*errl.data(),*/ childl.data(), YY.data(), YY.data() + nnodes + 1, NNODES,
      n, maxdepthd.data(), bottomd.data(), radiusd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(TreeBuildingKernel_time);

    START_TIMER;
    BH::ClearKernel2<<<blocks * 1, 1024, 0, stream>>>(
      startl.data(), massl.data(), NNODES, bottomd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(ClearKernel2_time);

    START_TIMER;
    BH::SummarizationKernel<<<blocks * FACTOR3, THREADS3, 0, stream>>>(
      countl.data(), childl.data(), massl.data(), YY.data(),
      YY.data() + nnodes + 1, NNODES, n, bottomd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(SummarizationKernel_time);

    START_TIMER;
    BH::SortKernel<<<blocks * FACTOR4, THREADS4, 0, stream>>>(
      sortl.data(), countl.data(), startl.data(), childl.data(), NNODES, n,
      bottomd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(SortKernel_time);

    START_TIMER;
    BH::RepulsionKernel<<<blocks * FACTOR5, THREADS5, 0, stream>>>(
      /*errl.data(),*/ theta, epssq, sortl.data(), childl.data(), massl.data(),
      YY.data(), YY.data() + nnodes + 1, rep_forces.data(),
      rep_forces.data() + nnodes + 1, Z_norm.data(), theta_squared, NNODES,
      FOUR_NNODES, n, radiusd_squared.data(), maxdepthd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(RepulsionTime);

    START_TIMER;
    BH::Find_Normalization<<<1, 1, 0, stream>>>(Z_norm.data(), n);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(Reduction_time);

    START_TIMER;
    // TODO: Calculate Kullback-Leibler divergence
    // For general embedding dimensions
    BH::attractive_kernel_bh<<<raft::ceildiv(NNZ, (value_idx)1024), 1024, 0,
                               stream>>>(
      VAL, COL, ROW, YY.data(), YY.data() + nnodes + 1, attr_forces.data(),
      attr_forces.data() + n, NNZ);
    CUDA_CHECK(cudaPeekAtLastError());
    END_TIMER(attractive_time);

    START_TIMER;
    BH::IntegrationKernel<<<blocks * FACTOR6, THREADS6, 0, stream>>>(
      learning_rate, momentum, early_exaggeration, YY.data(),
      YY.data() + nnodes + 1, attr_forces.data(), attr_forces.data() + n,
      rep_forces.data(), rep_forces.data() + nnodes + 1, gains_bh.data(),
      gains_bh.data() + n, old_forces.data(), old_forces.data() + n,
      Z_norm.data(), n);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(IntegrationKernel_time);
  }
  PRINT_TIMES;

  // Copy final YY into true output Y
  raft::copy(Y, YY.data(), n, stream);
  raft::copy(Y + n, YY.data() + nnodes + 1, n, stream);
}

}  // namespace TSNE
}  // namespace ML
