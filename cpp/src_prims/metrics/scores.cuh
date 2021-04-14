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
#include <linalg/power.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/stats/mean.cuh>

#include <memory>

#include <cuml/common/cuml_allocator.hpp>

#include <distance/distance.cuh>
#include <raft/spatial/knn/knn.hpp>
#include <selection/columnWiseSort.cuh>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define N_THREADS 512

namespace MLCommon {
namespace Score {

/**
 * @brief Compute a the rank of trustworthiness score
 * @param[in] ind_X: indexes given by pairwise distance and sorting
 * @param[in] ind_X_embedded: indexes given by KNN
 * @param[in] n: Number of samples
 * @param[in] n_neighbors: Number of neighbors considered by trustworthiness score
 * @param[in] work: Batch to consider (to do it at once use n * n_neighbors)
 * @param[out] rank: Resulting rank
 */
template <typename math_t, typename knn_index_t>
__global__ void compute_rank(math_t *ind_X, knn_index_t *ind_X_embedded, int n,
                             int n_neighbors, int work, double *rank) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= work) return;

  int n_idx = i / n_neighbors;
  int nn_idx = (i % n_neighbors) + 1;

  knn_index_t idx = ind_X_embedded[n_idx * (n_neighbors + 1) + nn_idx];
  math_t *sample_i = &ind_X[n_idx * n];

  // TODO: This could probably be binary searched, based on
  // the distances, as well. (re: https://github.com/rapidsai/cuml/issues/1698)
  for (int r = 1; r < n; r++) {
    if (sample_i[r] == idx) {
      int tmp = r - n_neighbors;
      if (tmp > 0) raft::myAtomicAdd<double>(rank, tmp);
      break;
    }
  }
}

/**
 * @brief Compute a kNN and returns the indices of the nearest neighbors
 * @param input Input matrix holding the dataset
 * @param n Number of samples
 * @param d Number of features
 * @param n_neighbors number of neighbors
 * @param d_alloc the device allocator to use for temp device memory
 * @param stream cuda stream to use
 * @return Matrix holding the indices of the nearest neighbors
 */
template <typename math_t>
long *get_knn_indices(const raft::handle_t &h, math_t *input, int n, int d,
                      int n_neighbors) {
  cudaStream_t stream = h.get_stream();
  auto d_alloc = h.get_device_allocator();

  long *d_pred_I =
    (int64_t *)d_alloc->allocate(n * n_neighbors * sizeof(int64_t), stream);
  math_t *d_pred_D =
    (math_t *)d_alloc->allocate(n * n_neighbors * sizeof(math_t), stream);

  std::vector<float *> ptrs(1);
  std::vector<int> sizes(1);
  ptrs[0] = input;
  sizes[0] = n;

  raft::spatial::knn::brute_force_knn(h, ptrs, sizes, d, input, n, d_pred_I,
                                      d_pred_D, n_neighbors);

  d_alloc->deallocate(d_pred_D, n * n_neighbors * sizeof(math_t), stream);
  return d_pred_I;
}

/**
 * @brief Compute the trustworthiness score
 * @tparam distance_type: Distance type to consider
 * @param X: Data in original dimension
 * @param X_embedded: Data in target dimension (embedding)
 * @param n: Number of samples
 * @param m: Number of features in high/original dimension
 * @param d: Number of features in low/embedded dimension
 * @param n_neighbors Number of neighbors considered by trustworthiness score
 * @param d_alloc device allocator to use for temp device memory
 * @param stream the cuda stream to use
 * @param batchSize batch size
 * @return Trustworthiness score
 */
template <typename math_t, raft::distance::DistanceType distance_type>
double trustworthiness_score(const raft::handle_t &h, math_t *X,
                             math_t *X_embedded, int n, int m, int d,
                             int n_neighbors, int batchSize = 512) {
  const int TMP_SIZE = batchSize * n;

  cudaStream_t stream = h.get_stream();
  auto d_alloc = h.get_device_allocator();

  typedef cutlass::Shape<8, 128, 128> OutputTile_t;

  math_t *d_pdist_tmp =
    (math_t *)d_alloc->allocate(TMP_SIZE * sizeof(math_t), stream);
  int *d_ind_X_tmp = (int *)d_alloc->allocate(TMP_SIZE * sizeof(int), stream);

  int64_t *ind_X_embedded =
    get_knn_indices(h, X_embedded, n, d, n_neighbors + 1);

  double t_tmp = 0.0;
  double t = 0.0;
  double *d_t = (double *)d_alloc->allocate(sizeof(double), stream);

  int toDo = n;
  while (toDo > 0) {
    int curBatchSize = min(toDo, batchSize);

    // Takes at most batchSize vectors at a time

    size_t workspaceSize = 0;

    MLCommon::Distance::distance<distance_type, math_t, math_t, math_t,
                                 OutputTile_t>(
      &X[(n - toDo) * m], X, d_pdist_tmp, curBatchSize, n, m, (void *)nullptr,
      workspaceSize, stream);
    CUDA_CHECK(cudaPeekAtLastError());

    size_t colSortWorkspaceSize = 0;
    bool bAllocWorkspace = false;
    char *sortColsWorkspace;

    MLCommon::Selection::sortColumnsPerRow(
      d_pdist_tmp, d_ind_X_tmp, curBatchSize, n, bAllocWorkspace, nullptr,
      colSortWorkspaceSize, stream);

    if (bAllocWorkspace) {
      sortColsWorkspace =
        (char *)d_alloc->allocate(colSortWorkspaceSize, stream);

      MLCommon::Selection::sortColumnsPerRow(
        d_pdist_tmp, d_ind_X_tmp, curBatchSize, n, bAllocWorkspace,
        sortColsWorkspace, colSortWorkspaceSize, stream);
    }
    CUDA_CHECK(cudaPeekAtLastError());

    t_tmp = 0.0;
    raft::update_device(d_t, &t_tmp, 1, stream);

    int work = curBatchSize * n_neighbors;
    int n_blocks = raft::ceildiv(work, N_THREADS);
    compute_rank<<<n_blocks, N_THREADS, 0, stream>>>(
      d_ind_X_tmp, &ind_X_embedded[(n - toDo) * (n_neighbors + 1)], n,
      n_neighbors, curBatchSize * n_neighbors, d_t);
    CUDA_CHECK(cudaPeekAtLastError());

    raft::update_host(&t_tmp, d_t, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (bAllocWorkspace) {
      d_alloc->deallocate(sortColsWorkspace, colSortWorkspaceSize, stream);
    }

    t += t_tmp;

    toDo -= curBatchSize;
  }

  t =
    1.0 -
    ((2.0 / ((n * n_neighbors) * ((2.0 * n) - (3.0 * n_neighbors) - 1.0))) * t);

  d_alloc->deallocate(ind_X_embedded, n * (n_neighbors + 1) * sizeof(int64_t),
                      stream);
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
 * @param stream: cuda stream
 * @return: The R-squared value.
 */
template <typename math_t>
math_t r2_score(math_t *y, math_t *y_hat, int n, cudaStream_t stream) {
  math_t *y_bar;
  raft::allocate(y_bar, 1);

  raft::stats::mean(y_bar, y, 1, n, false, false, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  math_t *sse_arr;
  raft::allocate(sse_arr, n);

  raft::linalg::eltwiseSub(sse_arr, y, y_hat, n, stream);
  MLCommon::LinAlg::powerScalar(sse_arr, sse_arr, math_t(2.0), n, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  math_t *ssto_arr;
  raft::allocate(ssto_arr, n);

  raft::linalg::subtractDevScalar(ssto_arr, y, y_bar, n, stream);
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
  raft::linalg::eltwiseSub(diffs_array, predictions, ref_predictions, n,
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
    raft::myAtomicAdd(&shmem[0], abs_diff);
    raft::myAtomicAdd(&shmem[1], diff * diff);

    // update absolute difference in global memory for subsequent abs. median computation
    abs_diffs[i] = abs_diff;
  }
  __syncthreads();

  // Update tmp_sum w/ total abs_difference_sum and squared difference sum.
  for (int i = threadIdx.x; i < 2; i += blockDim.x) {
    raft::myAtomicAdd(&tmp_sums[i], shmem[i]);
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
  int block_cnt = raft::ceildiv(n, thread_cnt);

  int array_size = n * sizeof(double);
  double *abs_diffs_array = (double *)d_alloc->allocate(array_size, stream);
  double *sorted_abs_diffs = (double *)d_alloc->allocate(array_size, stream);
  double *tmp_sums = (double *)d_alloc->allocate(2 * sizeof(double), stream);
  CUDA_CHECK(cudaMemsetAsync(tmp_sums, 0, 2 * sizeof(double), stream));

  reg_metrics_kernel<T><<<block_cnt, thread_cnt, 0, stream>>>(
    predictions, ref_predictions, n, abs_diffs_array, tmp_sums);
  CUDA_CHECK(cudaGetLastError());
  raft::update_host(&mean_errors[0], tmp_sums, 2, stream);
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

  raft::update_host(h_sorted_abs_diffs.data(), sorted_abs_diffs, n, stream);
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
