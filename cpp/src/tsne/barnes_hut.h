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

#define CUDA_CHECK_GOTO(call)                                            \
  do {                                                                   \
    cudaError_t status = call;                                           \
    if (status != cudaSuccess) {                                         \
      printf("ERROR: CUDA call='%s' at file=%s line=%d failed with %s ", \
             #call, __FILE__, __LINE__, cudaGetErrorString(status));     \
      goto ERROR;                                                        \
    }                                                                    \
  } while (0)


namespace ML {
namespace TSNE {

/**
 * @brief Fast Dimensionality reduction via TSNE using the Barnes Hut O(NlogN) approximation.
 * @input param VAL: The values in the attractive forces COO matrix.
 * @input param COL: The column indices in the attractive forces COO matrix.
 * @input param ROW: The row indices in the attractive forces COO matrix.
 * @input param NNZ: The number of non zeros in the attractive forces COO matrix.
 * @input param handle: The GPU handle.
 * @output param Y: The final embedding. Will overwrite this internally.
 * @input param n: Number of rows in data X.
 * @input param epssq: A tiny jitter to promote numerical stability.
 * @input param early_exaggeration: How much early pressure you want the clusters in TSNE to spread out more.
 * @input param exaggeration_iter: How many iterations you want the early pressure to run for.
 * @input param min_gain: Rounds up small gradient updates.
 * @input param pre_learning_rate: The learning rate during the exaggeration phase.
 * @input param post_learning_rate: The learning rate after the exaggeration phase.
 * @input param max_iter: The maximum number of iterations TSNE should run for.
 * @input param min_grad_norm: The smallest gradient norm TSNE should terminate on.
 * @input param pre_momentum: The momentum used during the exaggeration phase.
 * @input param post_momentum: The momentum used after the exaggeration phase.
 * @input param random_state: Set this to -1 for pure random intializations or >= 0 for reproducible outputs.
 * @input param verbose: Whether to print error messages or not.
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
                const long long random_state = -1, const bool verbose = true) {
  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  // Get device properites
  //---------------------------------------------------
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  const int blocks = deviceProp.multiProcessorCount;

  int nnodes = n * 2;
  if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
  while ((nnodes & (32 - 1)) != 0) nnodes++;
  nnodes--;
  if (verbose) printf("N_nodes = %d blocks = %d\n", nnodes, blocks);
  
  
  const int FOUR_NNODES = 4 * nnodes;
  const int FOUR_N = 4 * n;
  const float theta_squared = theta * theta;
  const int NNODES = nnodes;
  
  thrust::device_ptr<float> begin_massl;
  thrust::device_ptr<float> begin_gains_bh;
  
  float momentum = pre_momentum;
  float learning_rate = pre_learning_rate;
  
  // Allocate more space
  //---------------------------------------------------
  int *errl = NULL;
  unsigned int *limiter = NULL;
  int *maxdepthd = NULL;
  int *stepd = NULL;
  int *bottomd = NULL;
  float *radiusd = NULL;
  
  int *startl = NULL;
  int *childl = NULL;
  float *massl = NULL;
  float *maxxl = NULL;
  float *maxyl = NULL;
  float *minxl = NULL;
  float *minyl = NULL;
  int *countl = NULL;
  int *sortl = NULL;

  float *rep_forces = NULL;
  float *attr_forces = NULL;
  float *norm_add1 = NULL;
  float *norm = NULL;
  float *Z_norm = NULL;

  float *radiusd_squared = NULL;
  float *gains_bh = NULL;
  float *old_forces = NULL;
  float *YY = NULL;
  
  errl = (int *)d_alloc->allocate(sizeof(int), stream);
  limiter = (unsigned int *)d_alloc->allocate(sizeof(unsigned int), stream);
  maxdepthd = (int *)d_alloc->allocate(sizeof(int), stream);
  stepd = (int *)d_alloc->allocate(sizeof(int), stream);
  bottomd = (int *)d_alloc->allocate(sizeof(int), stream);
  radiusd = (float *)d_alloc->allocate(sizeof(float), stream);
  
  if (errl == NULL || limiter == NULL || maxdepthd == NULL ||
      stepd == NULL || bottomd == NULL || radiusd == NULL) {
    printf("[Error] Out of Memory");
    goto ERROR;
  }

  TSNE::InitializationKernel<<<1, 1, 0, stream>>>(errl, limiter, maxdepthd,
                                                  stepd, radiusd);
  CUDA_CHECK_GOTO(cudaPeekAtLastError());

  // Actual mallocs
  startl = (int *)d_alloc->allocate(sizeof(int) * (nnodes + 1), stream);
  childl = (int *)d_alloc->allocate(sizeof(int) * (nnodes + 1) * 4, stream);
  massl = (float *)d_alloc->allocate(sizeof(float) * (nnodes + 1), stream);
  
  if (startl == NULL || childl == NULL || massl == NULL) {
    printf("[Error] Out of Memory");
    goto ERROR;
  }
  begin_massl = thrust::device_pointer_cast(massl);
  thrust::fill(thrust::cuda::par.on(stream), begin_massl,
               begin_massl + (nnodes + 1), 1.0f);

  maxxl = (float *)d_alloc->allocate(sizeof(float) * blocks * FACTOR1, stream);
  maxyl = (float *)d_alloc->allocate(sizeof(float) * blocks * FACTOR1, stream);
  minxl = (float *)d_alloc->allocate(sizeof(float) * blocks * FACTOR1, stream);
  minyl = (float *)d_alloc->allocate(sizeof(float) * blocks * FACTOR1, stream);
  
  if (maxxl == NULL || maxyl == NULL || minxl == NULL || minyl == NULL) {
    printf("[Error] Out of Memory");
    goto ERROR;
  }

  // SummarizationKernel
  countl = (int *)d_alloc->allocate(sizeof(int) * (nnodes + 1), stream);

  // SortKernel
  sortl = (int *)d_alloc->allocate(sizeof(int) * (nnodes + 1), stream);
  
  if (countl == NULL || sortl == NULL) {
    printf("[Error] Out of Memory");
    goto ERROR;
  }

  // RepulsionKernel
  rep_forces = (float *)d_alloc->allocate(sizeof(float)*(nnodes + 1)*2, stream);
  attr_forces = (float *)d_alloc->allocate(sizeof(float)*n*2, stream);
  
  if (rep_forces == NULL || attr_forces == NULL) {
    printf("[Error] Out of Memory");
    goto ERROR;
  }

  norm_add1 = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  norm = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  Z_norm = (float *)d_alloc->allocate(sizeof(float), stream);
  
  if (norm_add1 == NULL || norm == NULL || Z_norm == NULL) {
    printf("[Error] Out of Memory");
    goto ERROR;
  }

  radiusd_squared = (float *)d_alloc->allocate(sizeof(float), stream);
  gains_bh = (float *)d_alloc->allocate(sizeof(float) * n * 2, stream);
  
  if (radiusd_squared == NULL || gains_bh == NULL) {
    printf("[Error] Out of Memory");
    goto ERROR;
  }
  begin_gains_bh = thrust::device_pointer_cast(gains_bh);
  thrust::fill(thrust::cuda::par.on(stream), begin_gains_bh,
               begin_gains_bh + (n * 2), 1.0f);

  old_forces = (float *)d_alloc->allocate(sizeof(float) * n * 2, stream);
  YY = (float *)d_alloc->allocate(sizeof(float) * (nnodes + 1) * 2, stream);
  
  if (old_forces == NULL || YY == NULL) {
    printf("[Error] Out of Memory");
    goto ERROR;
  }
  random_vector(YY, -100.0f, 100.0f, (nnodes + 1) * 2, stream, random_state);
  CUDA_CHECK_GOTO(cudaMemsetAsync(old_forces, 0, sizeof(float) * n * 2, stream));
  

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
  if (verbose) printf("[Info] Start gradient updates!\n");

  for (int iter = 0; iter < max_iter; iter++) {
    CUDA_CHECK_GOTO(
      cudaMemsetAsync(rep_forces, 0, sizeof(float) * (nnodes + 1) * 2, stream));
    CUDA_CHECK_GOTO(
      cudaMemsetAsync(attr_forces, 0, sizeof(float) * n * 2, stream));
    TSNE::Reset_Normalization<<<1, 1, 0, stream>>>(Z_norm, radiusd_squared,
                                                   bottomd, NNODES, radiusd);
    CUDA_CHECK(cudaPeekAtLastError());

    if (iter == exaggeration_iter) {
      momentum = post_momentum;
      // Divide perplexities
      const float div = 1.0f / early_exaggeration;
      MLCommon::LinAlg::scalarMultiply(VAL, VAL, div, NNZ, stream);
    }

    START_TIMER;
    TSNE::BoundingBoxKernel<<<blocks * FACTOR1, THREADS1, 0, stream>>>(
      startl, childl, massl, YY, YY + nnodes + 1, maxxl, maxyl, minxl, minyl,
      FOUR_NNODES, NNODES, n, limiter, stepd, radiusd);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    END_TIMER(BoundingBoxKernel_time);

    START_TIMER;
    TSNE::ClearKernel1<<<blocks, 1024, 0, stream>>>(childl, FOUR_NNODES,
                                                    FOUR_N);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    END_TIMER(ClearKernel1_time);

    START_TIMER;
    TSNE::TreeBuildingKernel<<<blocks * FACTOR2, THREADS2, 0, stream>>>(
      errl, childl, YY, YY + nnodes + 1, NNODES, n, maxdepthd, bottomd,
      radiusd);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    END_TIMER(TreeBuildingKernel_time);

    START_TIMER;
    TSNE::ClearKernel2<<<blocks * 1, 1024, 0, stream>>>(startl, massl, NNODES,
                                                        bottomd);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    END_TIMER(ClearKernel2_time);

    START_TIMER;
    TSNE::SummarizationKernel<<<blocks * FACTOR3, THREADS3, 0, stream>>>(
      countl, childl, massl, YY, YY + nnodes + 1, NNODES, n, bottomd);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    END_TIMER(SummarizationKernel_time);

    START_TIMER;
    TSNE::SortKernel<<<blocks * FACTOR4, THREADS4, 0, stream>>>(
      sortl, countl, startl, childl, NNODES, n, bottomd);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    END_TIMER(SortKernel_time);

    START_TIMER;
    TSNE::RepulsionKernel<<<blocks * FACTOR5, THREADS5, 0, stream>>>(
      errl, theta, epssq, sortl, childl, massl, YY, YY + nnodes + 1, rep_forces,
      rep_forces + nnodes + 1, Z_norm, theta_squared, FOUR_NNODES, n,
      radiusd_squared, maxdepthd, stepd);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    END_TIMER(RepulsionTime);

    START_TIMER;
    TSNE::Find_Normalization<<<1, 1, 0, stream>>>(Z_norm, (float)n);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    END_TIMER(Reduction_time);

    START_TIMER;
    TSNE::get_norm<<<MLCommon::ceildiv(n, 1024), 1024, 0, stream>>>(
      YY, YY + nnodes + 1, norm, norm_add1, n);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    // TODO: Calculate Kullback-Leibler divergence
    // For general embedding dimensions
    TSNE::
      attractive_kernel_bh<<<MLCommon::ceildiv(NNZ, 1024), 1024, 0, stream>>>(
        VAL, COL, ROW, YY, YY + nnodes + 1, norm, norm_add1, attr_forces,
        attr_forces + n, NNZ);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());
    END_TIMER(attractive_time);

    START_TIMER;
    TSNE::IntegrationKernel<<<blocks * FACTOR6, THREADS6, 0, stream>>>(
      learning_rate, momentum, early_exaggeration, YY, YY + nnodes + 1,
      attr_forces, attr_forces + n, rep_forces, rep_forces + nnodes + 1,
      gains_bh, gains_bh + n, old_forces, old_forces + n, Z_norm, n);
    CUDA_CHECK_GOTO(cudaPeekAtLastError());

    END_TIMER(IntegrationKernel_time);
  }
  PRINT_TIMES;

  // Copy final YY into true output Y
  thrust::device_ptr<float> Y_begin = thrust::device_pointer_cast(Y);
  thrust::copy(thrust::cuda::par.on(stream), YY, YY + n, Y_begin);
  thrust::copy(thrust::cuda::par.on(stream), YY + nnodes + 1,
               YY + nnodes + 1 + n, Y_begin + n);

  // Deallocate everything
  ERROR:
  if (errl != NULL)
    d_alloc->deallocate(errl, sizeof(int), stream);
  if (limiter != NULL)
    d_alloc->deallocate(limiter, sizeof(unsigned int), stream);
  if (maxdepthd != NULL)
    d_alloc->deallocate(maxdepthd, sizeof(int), stream);
  if (stepd != NULL)
    d_alloc->deallocate(stepd, sizeof(int), stream);
  if (bottomd != NULL)
    d_alloc->deallocate(bottomd, sizeof(int), stream);
  if (radiusd != NULL)
    d_alloc->deallocate(radiusd, sizeof(float), stream);
  
  if (startl != NULL)
    d_alloc->deallocate(startl, sizeof(int) * (nnodes + 1), stream);
  if (childl != NULL)
    d_alloc->deallocate(childl, sizeof(int) * (nnodes + 1) * 4, stream);
  if (massl != NULL)
    d_alloc->deallocate(massl, sizeof(float) * (nnodes + 1), stream);

  if (maxxl != NULL)
    d_alloc->deallocate(maxxl, sizeof(float) * blocks * FACTOR1, stream);
  if (maxyl != NULL)
    d_alloc->deallocate(maxyl, sizeof(float) * blocks * FACTOR1, stream);
  if (minxl != NULL)
    d_alloc->deallocate(minxl, sizeof(float) * blocks * FACTOR1, stream);
  if (minyl != NULL)
    d_alloc->deallocate(minyl, sizeof(float) * blocks * FACTOR1, stream);

  if (countl != NULL)
    d_alloc->deallocate(countl, sizeof(int) * (nnodes + 1), stream);
  if (sortl != NULL)
    d_alloc->deallocate(sortl, sizeof(int) * (nnodes + 1), stream);

  if (rep_forces != NULL)
    d_alloc->deallocate(rep_forces, sizeof(float) * (nnodes + 1) * 2, stream);
  if (attr_forces != NULL)
    d_alloc->deallocate(attr_forces, sizeof(float) * n * 2, stream);
  if (norm != NULL)
    d_alloc->deallocate(norm, sizeof(float) * n, stream);
  if (norm_add1 != NULL)
    d_alloc->deallocate(norm_add1, sizeof(float) * n, stream);

  if (Z_norm != NULL)
    d_alloc->deallocate(Z_norm, sizeof(float), stream);
  if (radiusd_squared != NULL)
    d_alloc->deallocate(radiusd_squared, sizeof(float), stream);

  if (gains_bh != NULL)
    d_alloc->deallocate(gains_bh, sizeof(float) * n * 2, stream);
  if (old_forces != NULL)
    d_alloc->deallocate(old_forces, sizeof(float) * n * 2, stream);

  if (YY != NULL)
    d_alloc->deallocate(YY, sizeof(float) * (nnodes + 1) * 2, stream);
}

}  // namespace TSNE
}  // namespace ML
