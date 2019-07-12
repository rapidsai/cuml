/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include "bh_kernels.h"
#include "utils.h"

namespace ML {
namespace TSNE {

void Barnes_Hut(float *VAL, const int *COL, const int *ROW, const int NNZ,
                const cumlHandle &handle, float *Y, const int n,
                const float theta = 0.5f, const float epssq = 0.0025,
                const float early_exaggeration = 12.0f,
                const int exaggeration_iter = 250, const float min_gain = 0.01f,
                const float pre_learning_rate = 200.0f,
                const float post_learning_rate = 500.0f,
                const int max_iter = 1000, const float min_grad_norm = 1e-7,
                const float pre_momentum = 0.5, const float post_momentum = 0.8,
                const long long random_state = -1, const bool verbose = true) {
  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  // Get device properites
  //---------------------------------------------------
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int blocks = deviceProp.multiProcessorCount;

  int nnodes = n * 2;
  if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
  while ((nnodes & (32 - 1)) != 0) nnodes++;
  nnodes--;
  if (verbose) printf("N_nodes = %d blocks = %d\n", nnodes, blocks);

  // Allocate more space
  //---------------------------------------------------
  int *errl = (int *)d_alloc->allocate(sizeof(int), stream);
  TSNE::InitializationKernel<<<1, 1, 0, stream>>>(errl);
  CUDA_CHECK(cudaPeekAtLastError());

  const int FOUR_NNODES_ = 4 * nnodes;
  CUDA_CHECK(cudaMemcpyToSymbolAsync(FOUR_NNODES, &FOUR_NNODES_, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, stream));
  const int FOUR_N_ = 4 * n;
  CUDA_CHECK(cudaMemcpyToSymbolAsync(FOUR_N, &FOUR_N_, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, stream));
  const float theta_squared_ = theta * theta;
  CUDA_CHECK(cudaMemcpyToSymbolAsync(theta_squared, &theta_squared_,
                                     sizeof(float), 0, cudaMemcpyHostToDevice,
                                     stream));

  CUDA_CHECK(cudaMemcpyToSymbolAsync(NNODES, &nnodes, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyToSymbolAsync(N, &n, sizeof(int), 0,
                                     cudaMemcpyHostToDevice, stream));

  // Actual mallocs
  int *startl = (int *)d_alloc->allocate(sizeof(int) * (nnodes + 1), stream);
  int *childl =
    (int *)d_alloc->allocate(sizeof(int) * (nnodes + 1) * 4, stream);
  float *massl =
    (float *)d_alloc->allocate(sizeof(float) * (nnodes + 1), stream);
  thrust::device_ptr<float> begin_massl = thrust::device_pointer_cast(massl);
  thrust::fill(thrust::cuda::par.on(stream), begin_massl,
               begin_massl + (nnodes + 1), 1.0f);

  float *maxxl =
    (float *)d_alloc->allocate(sizeof(float) * blocks * FACTOR1, stream);
  float *maxyl =
    (float *)d_alloc->allocate(sizeof(float) * blocks * FACTOR1, stream);
  float *minxl =
    (float *)d_alloc->allocate(sizeof(float) * blocks * FACTOR1, stream);
  float *minyl =
    (float *)d_alloc->allocate(sizeof(float) * blocks * FACTOR1, stream);

  // SummarizationKernel
  int *countl = (int *)d_alloc->allocate(sizeof(int) * (nnodes + 1), stream);

  // SortKernel
  int *sortl = (int *)d_alloc->allocate(sizeof(int) * (nnodes + 1), stream);

  // RepulsionKernel
  float *rep_forces =
    (float *)d_alloc->allocate(sizeof(float) * (nnodes + 1) * 2, stream);
  float *attr_forces = (float *)d_alloc->allocate(
    sizeof(float) * n * 2, stream);  // n*2 double for reduction sum

  //float *norml = (float *)d_alloc->allocate(sizeof(float) * (nnodes + 1), stream);
  float *norm_add1 = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  float *norm = (float *)d_alloc->allocate(sizeof(float) * n, stream);

  // Apply
  float *gains_bh = (float *)d_alloc->allocate(sizeof(float) * n * 2, stream);
  thrust::device_ptr<float> begin_gains_bh =
    thrust::device_pointer_cast(gains_bh);
  thrust::fill(thrust::cuda::par.on(stream), begin_gains_bh,
               begin_gains_bh + (n * 2), 1.0f);

  float *old_forces = (float *)d_alloc->allocate(sizeof(float) * n * 2, stream);
  CUDA_CHECK(cudaMemsetAsync(old_forces, 0, sizeof(float) * n * 2));

  float *YY =
    (float *)d_alloc->allocate(sizeof(float) * (nnodes + 1) * 2, stream);
  random_vector(YY, -100.0f, 100.0f, (nnodes + 1) * 2, stream, random_state);
  assert(YY != NULL && rep_forces != NULL);

  // Set cache levels for faster algorithm execution
  //---------------------------------------------------
  cudaFuncSetCacheConfig(TSNE::BoundingBoxKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TSNE::TreeBuildingKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::ClearKernel1, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::ClearKernel2, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::SummarizationKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TSNE::SortKernel, cudaFuncCachePreferL1);
#ifdef __KEPLER__
  cudaFuncSetCacheConfig(TSNE::RepulsionKernel, cudaFuncCachePreferEqual);
  cudaFuncSetCacheConfig(TSNE::attractive_kernel_bh, cudaFuncCachePreferEqual);
#else
  cudaFuncSetCacheConfig(TSNE::RepulsionKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(TSNE::attractive_kernel_bh, cudaFuncCachePreferL1);
#endif
  cudaFuncSetCacheConfig(TSNE::IntegrationKernel, cudaFuncCachePreferL1);

  //thrust::device_ptr<float> norml_begin = thrust::device_pointer_cast(norml);
  //thrust::device_ptr<float> norml_end = norml_begin + nnodes + 1;

  // Do gradient updates
  //---------------------------------------------------
  if (verbose) printf("[Info] Start gradient updates!\n");

  float momentum = pre_momentum;
  float learning_rate = pre_learning_rate;

  for (int iter = 0; iter < max_iter; iter++) {
    //printf("%d, ", iter);

    CUDA_CHECK(cudaMemset(rep_forces, 0, sizeof(float) * (nnodes + 1) * 2));
    CUDA_CHECK(cudaMemset(attr_forces, 0, sizeof(float) * n * 2));
    TSNE::Reset_Normalization<<<1, 1, 0, stream>>>();
    CUDA_CHECK(cudaPeekAtLastError());

    if (iter == exaggeration_iter) {
      momentum = post_momentum;
      // Divide perplexities
      const float div = 1.0f / early_exaggeration;
      MLCommon::LinAlg::scalarMultiply(VAL, VAL, div, NNZ, stream);
    }

    START_TIMER;
    TSNE::BoundingBoxKernel<<<blocks * FACTOR1, THREADS1, 0, stream>>>(
      startl, childl, massl, YY, YY + nnodes + 1, maxxl, maxyl, minxl, minyl);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(BoundingBoxKernel_time);

    START_TIMER;
    TSNE::ClearKernel1<<<blocks, 1024, 0, stream>>>(childl);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(ClearKernel1_time);

    START_TIMER;
    TSNE::TreeBuildingKernel<<<blocks * FACTOR2, THREADS2, 0, stream>>>(
      errl, childl, YY, YY + nnodes + 1);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(TreeBuildingKernel_time);

    START_TIMER;
    TSNE::ClearKernel2<<<blocks * 1, 1024, 0, stream>>>(startl, massl);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(ClearKernel2_time);

    START_TIMER;
    TSNE::SummarizationKernel<<<blocks * FACTOR3, THREADS3, 0, stream>>>(
      countl, childl, massl, YY, YY + nnodes + 1);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(SummarizationKernel_time);

    START_TIMER;
    TSNE::SortKernel<<<blocks * FACTOR4, THREADS4, 0, stream>>>(sortl, countl,
                                                                startl, childl);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(SortKernel_time);

    START_TIMER;
    TSNE::RepulsionKernel<<<blocks * FACTOR5, THREADS5, 0, stream>>>(
      errl, theta, epssq, sortl, childl, massl, YY, YY + nnodes + 1, rep_forces,
      rep_forces + nnodes + 1);
    //norml);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(RepulsionTime);

    START_TIMER;
    //const float Z = 1.0f / thrust::reduce(norml_begin, norml_end);
    //CUDA_CHECK(cudaMemcpyToSymbolAsync(Z_norm, &Z, sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    TSNE::Find_Normalization<<<1, 1, 0, stream>>>();
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(Reduction_time);

    START_TIMER;
    // Compute attractive forces
    TSNE::AttractiveKernel(VAL, COL, ROW, YY, nnodes, norm, norm_add1,
                           attr_forces, NNZ, n, stream);
    END_TIMER(attractive_time);

    START_TIMER;
    TSNE::IntegrationKernel<<<blocks * FACTOR6, THREADS6, 0, stream>>>(
      learning_rate, momentum, early_exaggeration, YY, YY + nnodes + 1,
      attr_forces, attr_forces + n, rep_forces, rep_forces + nnodes + 1,
      gains_bh, gains_bh + n, old_forces, old_forces + n);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(IntegrationKernel_time);
  }
  PRINT_TIMES;

  // Copy final YY into true output Y
  thrust::device_ptr<float> Y_begin = thrust::device_pointer_cast(Y);
  thrust::copy(YY, YY + n, Y_begin);
  thrust::copy(YY + nnodes + 1, YY + nnodes + 1 + n, Y_begin + n);

  // Deallocate everything
  d_alloc->deallocate(errl, sizeof(int), stream);
  d_alloc->deallocate(startl, sizeof(int) * (nnodes + 1), stream);
  d_alloc->deallocate(childl, sizeof(int) * (nnodes + 1) * 4, stream);
  d_alloc->deallocate(massl, sizeof(float) * (nnodes + 1), stream);

  d_alloc->deallocate(maxxl, sizeof(float) * blocks * FACTOR1, stream);
  d_alloc->deallocate(maxyl, sizeof(float) * blocks * FACTOR1, stream);
  d_alloc->deallocate(minxl, sizeof(float) * blocks * FACTOR1, stream);
  d_alloc->deallocate(minyl, sizeof(float) * blocks * FACTOR1, stream);

  d_alloc->deallocate(countl, sizeof(int) * (nnodes + 1), stream);
  d_alloc->deallocate(sortl, sizeof(int) * (nnodes + 1), stream);

  d_alloc->deallocate(rep_forces, sizeof(float) * (nnodes + 1) * 2, stream);
  d_alloc->deallocate(attr_forces, sizeof(float) * n * 2, stream);
  //d_alloc->deallocate(norml, sizeof(float) * (nnodes + 1), stream);
  d_alloc->deallocate(norm, sizeof(float) * n, stream);
  d_alloc->deallocate(norm_add1, sizeof(float) * n, stream);

  d_alloc->deallocate(gains_bh, sizeof(float) * n * 2, stream);
  d_alloc->deallocate(old_forces, sizeof(float) * n * 2, stream);

  d_alloc->deallocate(YY, sizeof(float) * (nnodes + 1) * 2, stream);
}

}  // namespace TSNE
}  // namespace ML