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

#include "linalg/eltwise.h"
#include "linalg/power.h"
#include "linalg/subtract.h"
#include "stats/mean.h"

#include <memory>

#include "common/cuml_allocator.hpp"

#include "selection/knn.h"
#include "distance/distance.h"
#include <selection/columnWiseSort.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define MAX_BATCH_SIZE 512
#define N_THREADS 512


namespace MLCommon {
    namespace Score {

      /**
      * @brief Compute a the rank of trustworthiness score
      * @input param ind_X: indexes given by pairwise distance and sorting
      * @input param ind_X_embedded: indexes given by KNN
      * @input param n: Number of samples
      * @input param n_neighbors: Number of neighbors considered by trustworthiness score
      * @input param work: Batch to consider (to do it at once use n * n_neighbors)
      * @output param rank: Resulting rank
      */
      template<typename math_t, typename knn_index_t>
      __global__ void compute_rank(math_t *ind_X, knn_index_t *ind_X_embedded,
                      int n, int n_neighbors, int work, double * rank)
      {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i >= work)
              return;

          int n_idx = i / n_neighbors;
          int nn_idx = (i % n_neighbors) + 1;

          knn_index_t idx = ind_X_embedded[n_idx * (n_neighbors+1) + nn_idx];
          math_t* sample_i = &ind_X[n_idx * n];
          for (int r = 1; r < n; r++)
          {
              if (sample_i[r] == idx)
              {
                  int tmp = r - n_neighbors;
                  if (tmp > 0)
                      atomicAdd(rank, tmp);
                  break;
              }
          }
      }


      /**
      * @brief Compute a kNN and returns the indexes of the nearest neighbors
      * @param input Input matrix holding the dataset
      * @param n Number of samples
      * @param d Number of features
      * @param d_alloc the device allocator to use for temp device memory
      * @param stream cuda stream to use
      * @return Matrix holding the indexes of the nearest neighbors
      */
      template<typename math_t>
      long* get_knn_indexes(math_t* input, int n,
                            int d, int n_neighbors,
                            std::shared_ptr<deviceAllocator> d_alloc,
                            cudaStream_t stream)
      {
          long* d_pred_I = (long*)d_alloc->allocate(n * n_neighbors * sizeof(long), stream);
          math_t* d_pred_D = (math_t*)d_alloc->allocate(n * n_neighbors * sizeof(math_t), stream);

          float **ptrs = new float*[1];
          ptrs[0] = input;

          int *sizes = new int[1];
          sizes[0] = n;

          MLCommon::Selection::brute_force_knn(ptrs, sizes, 1, d,
              input, n, d_pred_I, d_pred_D, n_neighbors, stream);

          d_alloc->deallocate(d_pred_D, n * n_neighbors * sizeof(math_t), stream);
          return d_pred_I;
      }

      /**
      * @brief Compute the trustworthiness score
      * @tparam distance_type: Distance type to consider
      * @param X: Data in original dimension
      * @param X_embedde: Data in target dimension (embedding)
      * @param n: Number of samples
      * @param m: Number of features in high/original dimension
      * @param d: Number of features in low/embedded dimension
      * @param n_neighbors Number of neighbors considered by trustworthiness score
      * @param d_alloc device allocator to use for temp device memory
      * @param stream the cuda stream to use
      * @return Trustworthiness score
      */
      template<typename math_t, Distance::DistanceType distance_type>
      double trustworthiness_score(math_t* X,
                          math_t* X_embedded, int n, int m, int d,
                          int n_neighbors,
                          std::shared_ptr<deviceAllocator> d_alloc,
                          cudaStream_t stream)
      {
          const int TMP_SIZE = MAX_BATCH_SIZE * n;

          size_t workspaceSize = 0; // EucUnexpandedL2Sqrt does not require workspace (may need change for other distances)
          typedef cutlass::Shape<8, 128, 128> OutputTile_t;
          bool bAllocWorkspace = false;

          math_t* d_pdist_tmp = (math_t*)d_alloc->allocate(TMP_SIZE * sizeof(math_t), stream);
          int* d_ind_X_tmp = (int*)d_alloc->allocate(TMP_SIZE * sizeof(int), stream);

          long* ind_X_embedded = get_knn_indexes(
              X_embedded,
              n, d, n_neighbors + 1,
              d_alloc, stream);

          double t_tmp = 0.0;
          double t = 0.0;
          double* d_t = (double*)d_alloc->allocate(sizeof(double), stream);

          int toDo = n;
          while (toDo > 0)
          {
              int batchSize = min(toDo, MAX_BATCH_SIZE);
              // Takes at most MAX_BATCH_SIZE vectors at a time

              MLCommon::Distance::distance<distance_type, math_t, math_t, math_t, OutputTile_t>
                      (&X[(n - toDo) * m], X,
                      d_pdist_tmp,
                      batchSize, n, m,
                      (void*)nullptr, workspaceSize,
                      stream
              );
              CUDA_CHECK(cudaPeekAtLastError());

              MLCommon::Selection::sortColumnsPerRow(d_pdist_tmp, d_ind_X_tmp,
                                  batchSize, n,
                                  bAllocWorkspace, NULL, workspaceSize,
                                  stream);
              CUDA_CHECK(cudaPeekAtLastError());

              t_tmp = 0.0;
              updateDevice(d_t, &t_tmp, 1, stream);

              int work = batchSize * n_neighbors;
              int n_blocks = work / N_THREADS + 1;
              compute_rank<<<n_blocks, N_THREADS, 0, stream>>>(d_ind_X_tmp,
                      &ind_X_embedded[(n - toDo) * (n_neighbors+1)],
                      n,
                      n_neighbors,
                      batchSize * n_neighbors,
                      d_t);
              CUDA_CHECK(cudaPeekAtLastError());

              updateHost(&t_tmp, d_t, 1, stream);
              t += t_tmp;

              toDo -= batchSize;
          }

          t = 1.0 - ((2.0 / ((n * n_neighbors) * ((2.0 * n) - (3.0 * n_neighbors) - 1.0))) * t);

          d_alloc->deallocate(ind_X_embedded, n * (n_neighbors + 1) * sizeof(long), stream);
          d_alloc->deallocate(d_pdist_tmp, TMP_SIZE * sizeof(math_t), stream);
          d_alloc->deallocate(d_ind_X_tmp, TMP_SIZE * sizeof(int), stream);
          d_alloc->deallocate(d_t, sizeof(double), stream);

          return t;
      }



/**
         * Calculates the "Coefficient of Determination" (R-Squared) score
         * normalizing the sum of squared errors by the total sum of squares.
         *
         * This score indicates the proportionate amount of variation in an
         * expected response variable is explained by the independent variables
         * in a linear regression model. The larger the R-squared value, the
         * more variability is explained by the linear regression model.
         *
         * @param y: Array of ground-truth response variables
         * @param y_hat: Array of predicted response variables
         * @param n: Number of elements in y and y_hat
         * @return: The R-squared value.
         */
template<typename math_t>
math_t r2_score(math_t *y, math_t *y_hat, int n, cudaStream_t stream) {
  math_t *y_bar;
  MLCommon::allocate(y_bar, 1);

  MLCommon::Stats::mean(y_bar, y, 1, n, false, false, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  math_t *sse_arr;
  MLCommon::allocate(sse_arr, n);

  MLCommon::LinAlg::eltwiseSub(sse_arr, y, y_hat, n, stream);
  MLCommon::LinAlg::powerScalar(sse_arr, sse_arr, math_t(2.0), n, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  math_t *ssto_arr;
  MLCommon::allocate(ssto_arr, n);

  MLCommon::LinAlg::subtractDevScalar(ssto_arr, y, y_bar, n, stream);
  MLCommon::LinAlg::powerScalar(ssto_arr, ssto_arr, math_t(2.0), n, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  thrust::device_ptr<math_t> d_sse = thrust::device_pointer_cast(sse_arr);
  thrust::device_ptr<math_t> d_ssto = thrust::device_pointer_cast(ssto_arr);

  math_t sse = thrust::reduce(thrust::cuda::par.on(stream), d_sse, d_sse + n);
  math_t ssto =
    thrust::reduce(thrust::cuda::par.on(stream), d_ssto, d_ssto + n);

  CUDA_CHECK(cudaFree(y_bar));
  CUDA_CHECK(cudaFree(sse_arr));
  CUDA_CHECK(cudaFree(ssto_arr));

  return 1.0 - sse / ssto;
}
}  // namespace Score
}  // namespace MLCommon
