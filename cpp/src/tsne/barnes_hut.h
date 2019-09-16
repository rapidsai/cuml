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
#include "common/device_buffer.hpp"

#define device_buffer MLCommon::device_buffer

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
 * @input param pca_intialization: Whether to intialize with PCA.
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
                const long long random_state = -1, const bool verbose = true,
                const bool pca_intialization = false)
{
  float max_bounds = 100;
  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  // Get device properites
  //---------------------------------------------------
  const int blocks = MLCommon::getMultiProcessorCount();

  int nnodes = n * 2;
  if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
  while ((nnodes & (32 - 1)) != 0) nnodes++;
  nnodes--;
  if (verbose) printf("N_nodes = %d blocks = %d\n", nnodes, blocks);


  const int FOUR_NNODES = 4 * nnodes;
  const int FOUR_N = 4 * n;
  const float theta_squared = theta * theta;
  const int NNODES = nnodes;
  const float N_float = n;
  const float div_N = 1.0f / N_float;

  // Allocate more space
  //---------------------------------------------------
  device_buffer<unsigned> limiter_(d_alloc, stream, 1);             unsigned *limiter = limiter_.data();
  CUDA_CHECK(cudaMemsetAsync(limiter, 0, sizeof(unsigned), stream));

  device_buffer<int> maxdepthd_(d_alloc, stream, 1);                int *maxdepthd = maxdepthd_.data();
  thrust::fill(thrust::cuda::par.on(stream), maxdepthd, maxdepthd + 1, 1);

  device_buffer<int> bottomd_(d_alloc, stream, 1);                  int *bottomd = bottomd_.data();
  device_buffer<float> radiusd_(d_alloc, stream, 1);                float *radiusd = radiusd_.data();
  CUDA_CHECK(cudaMemsetAsync(radiusd, 0, sizeof(float), stream));

  // Actual mallocs
  device_buffer<int> startl_(d_alloc, stream, NNODES + 1);          int *startl = startl_.data();
  device_buffer<int> childl_(d_alloc, stream, (NNODES + 1)*4);      int *childl = childl_.data();
  device_buffer<float> massl_(d_alloc, stream, NNODES + 1);         float *massl = massl_.data();
  thrust::fill(thrust::cuda::par.on(stream), massl, massl + (NNODES + 1), 1.0f);

  device_buffer<float> maxxl_(d_alloc, stream, blocks*FACTOR1);     float *maxxl = maxxl_.data();
  device_buffer<float> maxyl_(d_alloc, stream, blocks*FACTOR1);     float *maxyl = maxyl_.data();
  device_buffer<float> minxl_(d_alloc, stream, blocks*FACTOR1);     float *minxl = minxl_.data();
  device_buffer<float> minyl_(d_alloc, stream, blocks*FACTOR1);     float *minyl = minyl_.data();

  // SummarizationKernel
  device_buffer<int> countl_(d_alloc, stream, NNODES + 1);          int *countl = countl_.data();

  // SortKernel
  device_buffer<int> sortl_(d_alloc, stream, NNODES + 1);           int *sortl = sortl_.data();

  // RepulsionKernel
  device_buffer<float> rep_forces_(d_alloc, stream, (NNODES+1)*2);  float *rep_forces = rep_forces_.data();
  float *attr_forces = Y; // Reuse embeddings to save n*2 memory

  // Normalizations
  device_buffer<float> norm_add1_(d_alloc, stream, n);              float *norm_add1 = norm_add1_.data();
  device_buffer<float> norm_(d_alloc, stream, n);                   float *norm = norm_.data();
  device_buffer<float> Z_norm_(d_alloc, stream, 1);                 float *Z_norm = Z_norm_.data();
  device_buffer<float> sums_(d_alloc, stream, 2);                   float *sums = sums_.data();

  device_buffer<float> radiusd_squared_(d_alloc, stream, 1);        float *radiusd_squared = radiusd_squared_.data();

  // Apply
  device_buffer<float> gains_bh_(d_alloc, stream, n*2);             float *gains_bh = gains_bh_.data();
  thrust::fill(thrust::cuda::par.on(stream), gains_bh, gains_bh + (n * 2), 1.0f);

  device_buffer<float> old_forces_(d_alloc, stream, n*2);           float *old_forces = old_forces_.data();
  CUDA_CHECK(cudaMemsetAsync(old_forces, 0, sizeof(float) * n * 2, stream));

  device_buffer<float> YY_(d_alloc, stream, (NNODES+1)*2);          float *YY = YY_.data();
  ASSERT(YY != NULL && rep_forces != NULL, "[ERROR] Possibly no more memory");


  // Intialize embeddings
  if (pca_intialization == true)
  {
    // Copy Y into YY
    MLCommon::copyAsync(YY, Y, n, stream);
    MLCommon::copyAsync(YY + NNODES + 1, Y + n, n, stream);
  }
  else
  {
    random_vector(YY, -0.001f, 0.001f, (NNODES + 1) * 2, stream, random_state);
  }


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
  cudaFuncSetCacheConfig(TSNE::mean_centre, cudaFuncCachePreferL1);

  // Do gradient updates
  //---------------------------------------------------
  if (verbose) printf("[Info] Start gradient updates!\n");

  float momentum = pre_momentum;
  float learning_rate = pre_learning_rate;

  for (int iter = 0; iter < max_iter; iter++)
  {
    if (verbose) printf("[Iter %d] ", iter);

    if (verbose) printf("Reset >");

    // Reset everything
    CUDA_CHECK(cudaMemsetAsync(attr_forces, 0, sizeof(float) * n * 2, stream));
    CUDA_CHECK(cudaMemsetAsync(rep_forces, 0, sizeof(float) * (NNODES + 1) * 2, stream));
    CUDA_CHECK(cudaMemsetAsync(Z_norm, 0, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(startl + NNODES, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(sums, 0, sizeof(float) * 2, stream));

    thrust::fill(thrust::cuda::par.on(stream), bottomd, bottomd + 1, NNODES);
    thrust::fill(thrust::cuda::par.on(stream), massl + NNODES, massl + NNODES + 1, -1.0f);

    if (iter == exaggeration_iter)
    {
      momentum = post_momentum;
      // Divide perplexities
      MLCommon::LinAlg::scalarMultiply(VAL, VAL, 1.0f / early_exaggeration, NNZ, stream);
    }


    if (verbose) printf("Bounding Box >");
    START_TIMER;
    TSNE::BoundingBoxKernel<<<blocks * FACTOR1, THREADS1, 0, stream>>>(
      /*startl,*/ childl, /* massl, */
      YY, YY + NNODES + 1,
      YY + NNODES, YY + 2*NNODES + 1,
      maxxl, maxyl, minxl, minyl,
      FOUR_NNODES, NNODES, n, limiter, radiusd);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(BoundingBoxKernel_time);


    if (verbose) printf("Clear >");
    START_TIMER;
    TSNE::ClearKernel1<<<blocks, 1024, 0, stream>>>(childl, FOUR_NNODES, FOUR_N);
    CUDA_CHECK(cudaPeekAtLastError());
    END_TIMER(ClearKernel1_time);


    if (verbose) printf("Tree Building >");
    START_TIMER;
    TSNE::TreeBuildingKernel<<<blocks * FACTOR2, THREADS2, 0, stream>>>(
      /*errl,*/ childl, YY, YY + NNODES + 1, NNODES, n, maxdepthd, bottomd, radiusd);
    CUDA_CHECK(cudaPeekAtLastError());
    END_TIMER(TreeBuildingKernel_time);


    if (verbose) printf("Clear >");
    START_TIMER;
    TSNE::ClearKernel2<<<blocks, 1024, 0, stream>>>(startl, massl, NNODES, bottomd);
    CUDA_CHECK(cudaPeekAtLastError());
    END_TIMER(ClearKernel2_time);


    if (verbose) printf("Summarization >");
    START_TIMER;
    TSNE::SummarizationKernel<<<blocks * FACTOR3, THREADS3, 0, stream>>>(
      countl, childl, massl, YY, YY + NNODES + 1, NNODES, n, bottomd);
    CUDA_CHECK(cudaPeekAtLastError());
    END_TIMER(SummarizationKernel_time);


    if (verbose) printf("Sort >");
    START_TIMER;
    TSNE::SortKernel<<<blocks * FACTOR4, THREADS4, 0, stream>>>(
      sortl, countl, startl, childl, NNODES, n, bottomd);
    CUDA_CHECK(cudaPeekAtLastError());
    END_TIMER(SortKernel_time);


    if (verbose) printf("Repulsion >");
    START_TIMER;

    // Find radius^2
    MLCommon::LinAlg::unaryOp(radiusd_squared, radiusd, 1, [] __device__(float x) {return x*x;}, stream);

    TSNE::RepulsionKernel<<<blocks * FACTOR5, THREADS5, 0, stream>>>(
      /*errl,*/ theta, epssq, sortl, childl, massl,
      YY, YY + NNODES + 1,
      rep_forces, rep_forces + NNODES + 1,
      Z_norm, theta_squared, NNODES, FOUR_NNODES, n,
      radiusd_squared, maxdepthd);
    CUDA_CHECK(cudaPeekAtLastError());
    END_TIMER(RepulsionTime);


    if (verbose) printf("Norm >");
    START_TIMER;
    // Find normalization
    MLCommon::LinAlg::unaryOp(Z_norm, Z_norm, 1, [N_float] __device__(float x) {return 1.0f/(x - N_float);}, stream);
    END_TIMER(Reduction_time);


    START_TIMER;
    TSNE::get_norm<<<MLCommon::ceildiv(n, 1024), 1024, 0, stream>>>(
      YY, YY + NNODES + 1, norm, norm_add1, n);
    CUDA_CHECK(cudaPeekAtLastError());

    // TODO: Calculate Kullback-Leibler divergence
    // For general embedding dimensions
    if (verbose) printf("Attraction >");
    TSNE::attractive_kernel_bh<<<MLCommon::ceildiv(NNZ, 1024), 1024, 0, stream>>>(
        VAL, COL, ROW, YY, YY + NNODES + 1, norm, norm_add1, attr_forces,
        attr_forces + n, NNZ);
    CUDA_CHECK(cudaPeekAtLastError());
    END_TIMER(attractive_time);


    if (verbose) printf("Integration >");
    START_TIMER;
    TSNE::IntegrationKernel<<<blocks * FACTOR6, THREADS6, 0, stream>>>(
      learning_rate, momentum, early_exaggeration, YY, YY + NNODES + 1,
      attr_forces, attr_forces + n, rep_forces, rep_forces + NNODES + 1,
      gains_bh, gains_bh + n, old_forces, old_forces + n, Z_norm, n,
      max_bounds, sums);
    CUDA_CHECK(cudaPeekAtLastError());

    // Mean centre components
    MLCommon::LinAlg::unaryOp(sums, sums, 2, [div_N] __device__(float x) {return x * div_N;}, stream);
    TSNE::mean_centre<<<MLCommon::ceildiv(n, 1024), 1024, 0, stream>>>(YY, YY + NNODES + 1, sums, n);
    CUDA_CHECK(cudaPeekAtLastError());

    END_TIMER(IntegrationKernel_time);

    if (verbose) printf(" ...\n");

    max_bounds += 0.01f;
  }
  PRINT_TIMES;

  // Copy final YY into true output Y
  MLCommon::copyAsync(Y, YY, n, stream);
  MLCommon::copyAsync(Y + n, YY + NNODES + 1, n, stream);
}

}  // namespace TSNE
}  // namespace ML
