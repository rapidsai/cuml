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

#include <selection/columnWiseSort.h>
#include "distance/distance.h"
#include "selection/knn.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define INIT_COUNTER \
  int counter = 0;

#define BOUNDSCHECK(x, max) \
  if ((x) >= (max) || (x) < 0) { \
      printf("Bounds %d = [%d,%d]", __LINE__, x, (max)); \
      printf("Bounds %d = [%d,%d]", __LINE__, x, (max)); \
  } \
  if (counter++ >= 10000000) { \
      printf("Repeat %d", __LINE__); \
      printf("Repeat %d", __LINE__); \
  }

#define MAX_BATCH_SIZE 512
#define N_THREADS 512

namespace MLCommon {
namespace Score {

/**
 * @brief Compute a the rank of trustworthiness score
 * @input param ind_X: indexes given by pairwise distance and sorting
 * @input param embedded_indices: indexes given by KNN
 * @input param n: Number of samples
 * @input param n_neighbors: Number of neighbors considered by trustworthiness score
 * @input param work: Batch to consider (to do it at once use n * n_neighbors)
 * @output param rank: Resulting rank
 */
template <typename math_t, typename knn_index_t>
__global__ void
compute_rank(const math_t *__restrict ind_X,
             const knn_index_t *__restrict embedded_indices,
             const int n,
             const int n_neighbors,
             const int work,
             double *__restrict rank)
{
  INIT_COUNTER;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= work) return;

  const int n_idx = i / n_neighbors;
  const int nn_idx = (i % n_neighbors) + 1;

  BOUNDSCHECK(n_idx * (n_neighbors + 1) + nn_idx,   n * (n_neighbors + 1));
  const knn_index_t idx = embedded_indices[n_idx * (n_neighbors + 1) + nn_idx];

  BOUNDSCHECK(n_idx * n,   MAX_BATCH_SIZE * n);
  const math_t *__restrict sample_i = &ind_X[n_idx * n];
  for (int r = 1; r < n; r++)
  {
    if (sample_i[r] == idx)
    {
      const int tmp = r - n_neighbors;
      if (tmp > 0) atomicAdd(rank, tmp);
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
template <typename math_t>
long *__restrict
get_knn_indexes(const math_t *__restrict input,
                const int n,
                const int d,
                const int n_neighbors,
                std::shared_ptr<deviceAllocator> d_alloc,
                cudaStream_t stream)
{
  long *indices = (long *)d_alloc->allocate(n * n_neighbors * sizeof(long), stream);
  ASSERT(indices != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  math_t *distances = (math_t *)d_alloc->allocate(n * n_neighbors * sizeof(math_t), stream);
  ASSERT(distances != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  float **knn_input = new float *[1];
  int *sizes = new int[1];
  ASSERT(knn_input != NULL and sizes != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  knn_input[0] = (math_t*) input;
  sizes[0] = n;

  MLCommon::Selection::brute_force_knn(knn_input, sizes, 1, d,
                                       const_cast<float *>(input), n, indices,
                                       distances, n_neighbors, stream);

  d_alloc->deallocate(distances, n * n_neighbors * sizeof(math_t), stream);
  return indices;
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
template <typename math_t, Distance::DistanceType distance_type>
double
trustworthiness_score(const math_t *__restrict X,
                      const math_t *__restrict X_embedded,
                      const int n,
                      const int m,
                      const int d,
                      const int n_neighbors,
                      std::shared_ptr<deviceAllocator> d_alloc,
                      cudaStream_t stream)
{
  const int TMP_SIZE = MAX_BATCH_SIZE * n;

  typedef cutlass::Shape<8, 128, 128> OutputTile_t;

  math_t *distances = (math_t *)d_alloc->allocate(TMP_SIZE * sizeof(math_t), stream);
  ASSERT(distances != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  int *indices = (int *)d_alloc->allocate(TMP_SIZE * sizeof(int), stream);
  ASSERT(indices != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  long *embedded_indices = (long*) get_knn_indexes(X_embedded, n, d, n_neighbors + 1, d_alloc, stream);
  ASSERT(embedded_indices != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  double *d_t = (double *) d_alloc->allocate(sizeof(double), stream);
  ASSERT(d_t != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  int toDo = n;
  double t = 0.0;


  while (toDo > 0)
  {
    // Takes at most MAX_BATCH_SIZE vectors at a time
    const int batchSize = min(toDo, MAX_BATCH_SIZE);

    
    // Determine distance workspace size
    char *distance_workspace = NULL;

    const size_t distance_workspace_size = \
      MLCommon::Distance::getWorkspaceSize<distance_type, math_t, math_t, math_t>(
        &X[(n - toDo) * m], X, batchSize, n, m);
    
    if (distance_workspace_size != 0) {
      distance_workspace = (char*) d_alloc->allocate(distance_workspace_size, stream);
      ASSERT(distance_workspace != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);
    }
    

    // Find distances
    MLCommon::Distance::distance<distance_type, math_t, math_t, math_t,
                                 OutputTile_t>(
      &X[(n - toDo) * m], X, distances, batchSize, n, m, (void*)distance_workspace,
      distance_workspace_size, stream);

    CUDA_CHECK(cudaPeekAtLastError());
    

    // Free workspace
    if (distance_workspace_size != 0) {
      d_alloc->deallocate(distance_workspace, distance_workspace_size, stream);
    }
    
    
    // Determine sort columns workspace
    bool need_workspace = false;
    size_t sort_workspace_size = 0;
    MLCommon::Selection::sortColumnsPerRow(distances, indices, batchSize,
                                           n, need_workspace, NULL,
                                           sort_workspace_size, stream);
    CUDA_CHECK(cudaPeekAtLastError());


    if (need_workspace)
    {
      char *sort_workspace = (char*) d_alloc->allocate(sort_workspace_size, stream);
      ASSERT(sort_workspace != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

      MLCommon::Selection::sortColumnsPerRow(distances, indices, batchSize,
                                             n, need_workspace, (void*)sort_workspace,
                                             sort_workspace_size, stream);
      
      CUDA_CHECK(cudaPeekAtLastError());

      d_alloc->deallocate(sort_workspace, sort_workspace_size, stream);
    }
    

    double t_tmp = 0.0;
    updateDevice(d_t, &t_tmp, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));


    const int work = batchSize * n_neighbors;
    const int n_blocks = work / N_THREADS + 1;

    compute_rank<<<n_blocks, N_THREADS, 0, stream>>>(
      indices, &embedded_indices[(n - toDo) * (n_neighbors + 1)], n,
      n_neighbors, batchSize * n_neighbors, d_t);

    CUDA_CHECK(cudaPeekAtLastError());


    updateHost(&t_tmp, d_t, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    t += t_tmp;

    toDo -= batchSize;
  }

  t =
    1.0 -
    ((2.0 / ((n * n_neighbors) * ((2.0 * n) - (3.0 * n_neighbors) - 1.0))) * t);

  d_alloc->deallocate(embedded_indices, n * (n_neighbors + 1) * sizeof(long), stream);
  d_alloc->deallocate(distances, TMP_SIZE * sizeof(math_t), stream);
  d_alloc->deallocate(indices, TMP_SIZE * sizeof(int), stream);
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
template <typename math_t>
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

/**
 * @brief Compute accuracy of predictions. Useful for classification.
 * @tparam math_t: data type for predictions (e.g., int for classification)
 * @param[in] predictions: array of predictions (GPU pointer).
 * @param[in] ref_predictions: array of reference (ground-truth) predictions (GPU pointer).
 * @param[in] n: number of elements in each of predictions, ref_predictions.
 * @param[in] d_alloc: device allocator.
 * @param[in] stream: cuda stream.
 * @return: Accuracy score in [0, 1]; higher is better.
 */
template <typename math_t>
float accuracy_score(const math_t *predictions, const math_t *ref_predictions,
                     int n, std::shared_ptr<deviceAllocator> d_alloc,
                     cudaStream_t stream) {
  unsigned long long correctly_predicted = 0ULL;
  math_t *diffs_array = (math_t *)d_alloc->allocate(n * sizeof(math_t), stream);

  //TODO could write a kernel instead
  MLCommon::LinAlg::eltwiseSub(diffs_array, predictions, ref_predictions, n,
                               stream);
  CUDA_CHECK(cudaGetLastError());
  correctly_predicted = thrust::count(thrust::cuda::par.on(stream), diffs_array,
                                      diffs_array + n, 0);
  d_alloc->deallocate(diffs_array, n * sizeof(math_t), stream);

  float accuracy = correctly_predicted * 1.0f / n;
  return accuracy;
}

template <typename T>
__global__ void reg_metrics_kernel(const T *predictions,
                                   const T *ref_predictions, int n,
                                   double *abs_diffs, double *tmp_sums) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double shmem[2];  // {abs_difference_sum, squared difference sum}

  for (int i = threadIdx.x; i < 2; i += blockDim.x) {
    shmem[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    double diff = predictions[i] - ref_predictions[i];
    double abs_diff = abs(diff);
    atomicAdd(&shmem[0], abs_diff);
    atomicAdd(&shmem[1], diff * diff);

    // update absolute difference in global memory for subsequent abs. median computation
    abs_diffs[i] = abs_diff;
  }
  __syncthreads();

  // Update tmp_sum w/ total abs_difference_sum and squared difference sum.
  for (int i = threadIdx.x; i < 2; i += blockDim.x) {
    atomicAdd(&tmp_sums[i], shmem[i]);
  }
}

/**
 * @brief Compute regression metrics mean absolute error, mean squared error, median absolute error
 * @tparam T: data type for predictions (e.g., float or double for regression).
 * @param[in] predictions: array of predictions (GPU pointer).
 * @param[in] ref_predictions: array of reference (ground-truth) predictions (GPU pointer).
 * @param[in] n: number of elements in each of predictions, ref_predictions. Should be > 0.
 * @param[in] d_alloc: device allocator.
 * @param[in] stream: cuda stream.
 * @param[out] mean_abs_error: Mean Absolute Error. Sum over n of (|predictions[i] - ref_predictions[i]|) / n.
 * @param[out] mean_squared_error: Mean Squared Error. Sum over n of ((predictions[i] - ref_predictions[i])^2) / n.
 * @param[out] median_abs_error: Median Absolute Error. Median of |predictions[i] - ref_predictions[i]| for i in [0, n).
 */
template <typename T>
void regression_metrics(const T *predictions, const T *ref_predictions, int n,
                        std::shared_ptr<deviceAllocator> d_alloc,
                        cudaStream_t stream, double &mean_abs_error,
                        double &mean_squared_error, double &median_abs_error) {
  std::vector<double> mean_errors(2);
  std::vector<double> h_sorted_abs_diffs(n);
  int thread_cnt = 256;
  int block_cnt = ceildiv(n, thread_cnt);

  int array_size = n * sizeof(double);
  double *abs_diffs_array = (double *)d_alloc->allocate(array_size, stream);
  double *sorted_abs_diffs = (double *)d_alloc->allocate(array_size, stream);
  double *tmp_sums = (double *)d_alloc->allocate(2 * sizeof(double), stream);
  CUDA_CHECK(cudaMemsetAsync(tmp_sums, 0, 2 * sizeof(double), stream));

  reg_metrics_kernel<T><<<block_cnt, thread_cnt, 0, stream>>>(
    predictions, ref_predictions, n, abs_diffs_array, tmp_sums);
  CUDA_CHECK(cudaGetLastError());
  MLCommon::updateHost(&mean_errors[0], tmp_sums, 2, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  mean_abs_error = mean_errors[0] / n;
  mean_squared_error = mean_errors[1] / n;

  // Compute median error. Sort diffs_array and pick median value
  char *temp_storage = nullptr;
  size_t temp_storage_bytes;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
    (void *)temp_storage, temp_storage_bytes, abs_diffs_array, sorted_abs_diffs,
    n, 0, 8 * sizeof(double), stream));
  temp_storage = (char *)d_alloc->allocate(temp_storage_bytes, stream);
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
    (void *)temp_storage, temp_storage_bytes, abs_diffs_array, sorted_abs_diffs,
    n, 0, 8 * sizeof(double), stream));

  MLCommon::updateHost(h_sorted_abs_diffs.data(), sorted_abs_diffs, n, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int middle = n / 2;
  if (n % 2 == 1) {
    median_abs_error = h_sorted_abs_diffs[middle];
  } else {
    median_abs_error =
      (h_sorted_abs_diffs[middle] + h_sorted_abs_diffs[middle - 1]) / 2;
  }

  d_alloc->deallocate(abs_diffs_array, array_size, stream);
  d_alloc->deallocate(sorted_abs_diffs, array_size, stream);
  d_alloc->deallocate(temp_storage, temp_storage_bytes, stream);
  d_alloc->deallocate(tmp_sums, 2 * sizeof(double), stream);
}
}  // namespace Score
}  // namespace MLCommon
