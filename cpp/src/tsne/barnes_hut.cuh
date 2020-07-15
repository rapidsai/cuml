/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <common/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <cuml/common/logger.hpp>
#include "bh_kernels.cuh"
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
 * @param[in] early_exaggeration: How much early pressure you want the clusters in TSNE to spread out more.
 * @param[in] exaggeration_iter: How many iterations you want the early pressure to run for.
 * @param[in] min_gain: Rounds up small gradient updates.
 * @param[in] pre_learning_rate: The learning rate during the exaggeration phase.
 * @param[in] post_learning_rate: The learning rate after the exaggeration phase.
 * @param[in] max_iter: The maximum number of iterations TSNE should run for.
 * @param[in] min_grad_norm: The smallest gradient norm TSNE should terminate on.
 * @param[in] pre_momentum: The momentum used during the exaggeration phase.
 * @param[in] post_momentum: The momentum used after the exaggeration phase.
 * @param[in] random_state: Set this to -1 for pure random intializations or >= 0 for reproducible outputs.
 */
void Barnes_Hut(float *VAL, const int *COL, const int *ROW, const int NNZ,
                const cumlHandle &handle, float *Y, const int n,
                const float theta = 0.5f, const float epssq = 0.0025,
                const float early_exaggeration = 12.0f,
                const int exaggeration_iter = 250, const float min_gain = 0.01f,
                const float pre_learning_rate = 200.0f,
                const float post_learning_rate = 500.0f,
                const int max_iter = 1000, const float min_grad_norm = 1e-7,
                const float pre_momentum = 0.5, const float post_momentum = 0.8,
                const long long random_state = -1) {
  using MLCommon::device_buffer;
  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  // Get device properites
  //---------------------------------------------------
  const int blocks = MLCommon::getMultiProcessorCount();

  int nnodes = n * 2;
  if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
  while ((nnodes & (32 - 1)) != 0) nnodes++;
  nnodes--;
  CUML_LOG_DEBUG("N_nodes = %d blocks = %d", nnodes, blocks);

  // Allocate more space
  //---------------------------------------------------
  //int *errl = (int *)d_alloc->allocate(sizeof(int), stream);
  device_buffer<unsigned> limiter(d_alloc, stream, 1);
  device_buffer<int> maxdepthd(d_alloc, stream, 1);
  device_buffer<int> bottomd(d_alloc, stream, 1);
  device_buffer<float> radiusd(d_alloc, stream, 1);

  TSNE::InitializationKernel<<<1, 1, 0, stream>>>(
    /*errl,*/ limiter.data(), maxdepthd.data(), radiusd.data());

  CUDA_CHECK(cudaPeekAtLastError());

  const int FOUR_NNODES = 4 * nnodes;
  const int FOUR_N = 4 * n;
  const float theta_squared = theta * theta;
  const int NNODES = nnodes;
  bool h_chk_nan;

  // Actual mallocs
  device_buffer<int> startl(d_alloc, stream, nnodes + 1);
  device_buffer<int> childl(d_alloc, stream, (nnodes + 1) * 4);
  device_buffer<float> massl(d_alloc, stream, nnodes + 1);
  thrust::device_ptr<float> begin_massl =
    thrust::device_pointer_cast(massl.data());
  thrust::fill(thrust::cuda::par.on(stream), begin_massl,
               begin_massl + (nnodes + 1), 1.0f);

  device_buffer<float> maxxl(d_alloc, stream, blocks * FACTOR1);
  device_buffer<float> maxyl(d_alloc, stream, blocks * FACTOR1);
  device_buffer<float> minxl(d_alloc, stream, blocks * FACTOR1);
  device_buffer<float> minyl(d_alloc, stream, blocks * FACTOR1);

  // SummarizationKernel
  device_buffer<int> countl(d_alloc, stream, nnodes + 1);

  // SortKernel
  device_buffer<int> sortl(d_alloc, stream, nnodes + 1);

  // RepulsionKernel
  device_buffer<float> rep_forces(d_alloc, stream, (nnodes + 1) * 2);
  device_buffer<float> attr_forces(d_alloc, stream, n * 2);

  device_buffer<float> norm_add1(d_alloc, stream, n);
  device_buffer<float> norm(d_alloc, stream, n);
  device_buffer<float> Z_norm(d_alloc, stream, 1);

  device_buffer<float> radiusd_squared(d_alloc, stream, 1);

  // Apply
  device_buffer<float> gains_bh(d_alloc, stream, n * 2);
  thrust::device_ptr<float> begin_gains_bh =
    thrust::device_pointer_cast(gains_bh.data());
  thrust::fill(thrust::cuda::par.on(stream), begin_gains_bh,
               begin_gains_bh + (n * 2), 1.0f);

  device_buffer<float> old_forces(d_alloc, stream, n * 2);
  CUDA_CHECK(
    cudaMemsetAsync(old_forces.data(), 0, sizeof(float) * n * 2, stream));

  device_buffer<float> YY(d_alloc, stream, (nnodes + 1) * 2);
  device_buffer<float> YY_prev(d_alloc, stream, (nnodes + 1) * 2);
  random_vector(YY.data(), -0.0001f, 0.0001f, (nnodes + 1) * 2, stream,
                random_state);
  ASSERT(
    YY.data() != NULL && YY_prev.data() != NULL && rep_forces.data() != NULL,
    "[ERROR] Possibly no more memory");

  // Set cache levels for faster algorithm execution
  //---------------------------------------------------
  cudaFuncSetCacheConfig(TSNE::BoundingBoxKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TSNE::TreeBuildingKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::ClearKernel1, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::ClearKernel2, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::SummarizationKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TSNE::SortKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::RepulsionKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::attractive_kernel_bh, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::IntegrationKernel, cudaFuncCachePreferL1);
  // Do gradient updates
  //---------------------------------------------------
  CUML_LOG_DEBUG("Start gradient updates!");

  float momentum = pre_momentum;
  float learning_rate = pre_learning_rate;

  for (int iter = 0; iter < max_iter; iter++) {
    CUDA_CHECK(cudaMemsetAsync(rep_forces.data(), 0,
                               sizeof(float) * (nnodes + 1) * 2, stream));
    CUDA_CHECK(
      cudaMemsetAsync(attr_forces.data(), 0, sizeof(float) * n * 2, stream));

    MLCommon::copy(YY_prev.data(), YY.data(), (nnodes + 1) * 2, stream);
    TSNE::Reset_Normalization<<<1, 1, 0, stream>>>(
      Z_norm.data(), radiusd_squared.data(), bottomd.data(), NNODES,
      radiusd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    if (iter == exaggeration_iter) {
      momentum = post_momentum;
      // Divide perplexities
      const float div = 1.0f / early_exaggeration;
      MLCommon::LinAlg::scalarMultiply(VAL, VAL, div, NNZ, stream);
    }

    START_TIMER;
    TSNE::BoundingBoxKernel<<<blocks * FACTOR1, THREADS1, 0, stream>>>(
      startl.data(), childl.data(), massl.data(), YY.data(),
      YY.data() + nnodes + 1, maxxl.data(), maxyl.data(), minxl.data(),
      minyl.data(), FOUR_NNODES, NNODES, n, limiter.data(), radiusd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(BoundingBoxKernel_time);

    START_TIMER;
    TSNE::ClearKernel1<<<blocks, 1024, 0, stream>>>(childl.data(), FOUR_NNODES,
                                                    FOUR_N);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(ClearKernel1_time);

    START_TIMER;
    TSNE::TreeBuildingKernel<<<blocks * FACTOR2, THREADS2, 0, stream>>>(
      /*errl,*/ childl.data(), YY.data(), YY.data() + nnodes + 1, NNODES, n,
      maxdepthd.data(), bottomd.data(), radiusd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(TreeBuildingKernel_time);

    START_TIMER;
    TSNE::ClearKernel2<<<blocks * 1, 1024, 0, stream>>>(
      startl.data(), massl.data(), NNODES, bottomd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(ClearKernel2_time);

    START_TIMER;
    TSNE::SummarizationKernel<<<blocks * FACTOR3, THREADS3, 0, stream>>>(
      countl.data(), childl.data(), massl.data(), YY.data(),
      YY.data() + nnodes + 1, NNODES, n, bottomd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(SummarizationKernel_time);

    START_TIMER;
    TSNE::SortKernel<<<blocks * FACTOR4, THREADS4, 0, stream>>>(
      sortl.data(), countl.data(), startl.data(), childl.data(), NNODES, n,
      bottomd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(SortKernel_time);

    START_TIMER;
    TSNE::RepulsionKernel<<<blocks * FACTOR5, THREADS5, 0, stream>>>(
      /*errl,*/ theta, epssq, sortl.data(), childl.data(), massl.data(),
      YY.data(), YY.data() + nnodes + 1, rep_forces.data(),
      rep_forces.data() + nnodes + 1, Z_norm.data(), theta_squared, NNODES,
      FOUR_NNODES, n, radiusd_squared.data(), maxdepthd.data());
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(RepulsionTime);

    START_TIMER;
    TSNE::Find_Normalization<<<1, 1, 0, stream>>>(Z_norm.data(), n);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(Reduction_time);

    START_TIMER;
    TSNE::get_norm<<<MLCommon::ceildiv(n, 1024), 1024, 0, stream>>>(
      YY.data(), YY.data() + nnodes + 1, norm.data(), norm_add1.data(), n);
    CUDA_CHECK(cudaPeekAtLastError());

    // TODO: Calculate Kullback-Leibler divergence
    // For general embedding dimensions
    TSNE::
      attractive_kernel_bh<<<MLCommon::ceildiv(NNZ, 1024), 1024, 0, stream>>>(
        VAL, COL, ROW, YY.data(), YY.data() + nnodes + 1, norm.data(),
        norm_add1.data(), attr_forces.data(), attr_forces.data() + n, NNZ);
    CUDA_CHECK(cudaPeekAtLastError());
    END_TIMER(attractive_time);

    START_TIMER;
    TSNE::IntegrationKernel<<<blocks * FACTOR6, THREADS6, 0, stream>>>(
      learning_rate, momentum, early_exaggeration, YY.data(),
      YY.data() + nnodes + 1, attr_forces.data(), attr_forces.data() + n,
      rep_forces.data(), rep_forces.data() + nnodes + 1, gains_bh.data(),
      gains_bh.data() + n, old_forces.data(), old_forces.data() + n,
      Z_norm.data(), n);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(IntegrationKernel_time);

    h_chk_nan = thrust::transform_reduce(
      thrust::cuda::par.on(stream), YY.data(), YY.data() + (nnodes + 1) * 2,
      NaNTestKernel(), 0, thrust::plus<bool>());
    if (h_chk_nan) {
      CUML_LOG_DEBUG(
        "NaN result detected during Barnes Hut iteration: %d, returning last "
        "known good positions.",
        iter);
      MLCommon::copy(YY.data(), YY_prev.data(), (nnodes + 1) * 2, stream);
      break;
    }
  }
  PRINT_TIMES;

  // Copy final YY into true output Y
  thrust::device_ptr<float> Y_begin = thrust::device_pointer_cast(Y);

  MLCommon::copy(Y, YY.data(), n, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  MLCommon::copy(Y + n, YY.data() + nnodes + 1, n, stream);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace TSNE
}  // namespace ML
