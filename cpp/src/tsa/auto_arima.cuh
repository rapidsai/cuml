/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <common/fast_int_div.cuh>

#include <cuml/common/utils.hpp>

#include <raft/core/interruptible.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace ML {
namespace TimeSeries {

struct BoolToIntFunctor {
  HDI int operator()(const bool& a) const { return static_cast<int>(a); }
};

/**
 * Helper to compute the cumulative sum of a boolean mask
 *
 * @param[in]  mask       Input boolean array
 * @param[out] cumul      Output cumulative sum
 * @param[in]  mask_size  Size of the arrays
 * @param[in]  stream     CUDA stream
 */
void cumulative_sum_helper(const bool* mask, int* cumul, int mask_size, cudaStream_t stream)
{
  BoolToIntFunctor conversion_op;
  thrust::transform_iterator<BoolToIntFunctor, const bool*, thrust::use_default, int> itr(
    mask, conversion_op);

  // Determine temporary storage size
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, itr, cumul, mask_size, stream);

  // Allocate temporary storage
  rmm::device_uvector<uint8_t> temp_storage(temp_storage_bytes, stream);
  void* d_temp_storage = (void*)temp_storage.data();

  // Execute the scan
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, itr, cumul, mask_size, stream);
}

/**
 * Batch division by mask step 1: build an index of the position of each series
 * in its new batch and measure the size of each sub-batch
 *
 * @param[in]  d_mask     Boolean mask
 * @param[out] d_index    Index of each series in its new batch
 * @param[in]  batch_size Batch size
 * @param[in]  stream     CUDA stream
 * @return The number of 'true' series in the mask
 */
inline int divide_by_mask_build_index(const bool* d_mask,
                                      int* d_index,
                                      int batch_size,
                                      cudaStream_t stream)
{
  // Inverse mask
  rmm::device_uvector<bool> inv_mask(batch_size, stream);
  thrust::transform(thrust::cuda::par.on(stream),
                    d_mask,
                    d_mask + batch_size,
                    inv_mask.data(),
                    thrust::logical_not<bool>());

  // Cumulative sum of the inverse mask
  rmm::device_uvector<int> index0(batch_size, stream);
  cumulative_sum_helper(inv_mask.data(), index0.data(), batch_size, stream);

  // Cumulative sum of the mask
  rmm::device_uvector<int> index1(batch_size, stream);
  cumulative_sum_helper(d_mask, index1.data(), batch_size, stream);

  // Combine both cumulative sums according to the mask and subtract 1
  const int* d_index0 = index0.data();
  const int* d_index1 = index1.data();
  auto counting       = thrust::make_counting_iterator(0);
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int i) {
      d_index[i] = (d_mask[i] ? d_index1[i] : d_index0[i]) - 1;
    });

  // Compute and return the number of true elements in the mask
  int true_elements;
  raft::update_host(&true_elements, index1.data() + batch_size - 1, 1, stream);
  raft::interruptible::synchronize(stream);
  return true_elements;
}

/**
 * Kernel for the batch division by mask
 *
 * @param[in]  d_in       Input batch
 * @param[in]  d_mask     Boolean mask
 * @param[in]  d_index    Index of each series in its new batch
 * @param[out] d_out0     The sub-batch for the 'false' members
 * @param[out] d_out1     The sub-batch for the 'true' members
 * @param[in]  n_obs      Number of data points per series
 */
template <typename DataT>
CUML_KERNEL void divide_by_mask_kernel(const DataT* d_in,
                                       const bool* d_mask,
                                       const int* d_index,
                                       DataT* d_out0,
                                       DataT* d_out1,
                                       int n_obs)
{
  const DataT* b_in = d_in + n_obs * blockIdx.x;
  DataT* b_out      = (d_mask[blockIdx.x] ? d_out1 : d_out0) + n_obs * d_index[blockIdx.x];

  for (int i = threadIdx.x; i < n_obs; i += blockDim.x) {
    b_out[i] = b_in[i];
  }
}

/**
 * Batch division by mask step 2: create both sub-batches from the mask and
 * index
 *
 * @param[in]  d_in       Input batch. Each series is a contiguous chunk
 * @param[in]  d_mask     Boolean mask
 * @param[in]  d_index    Index of each series in its new batch
 * @param[out] d_out0     The sub-batch for the 'false' members
 * @param[out] d_out1     The sub-batch for the 'true' members
 * @param[in]  batch_size Batch size
 * @param[in]  n_obs      Number of data points per series
 * @param[in]  stream     CUDA stream
 */
template <typename DataT>
inline void divide_by_mask_execute(const DataT* d_in,
                                   const bool* d_mask,
                                   const int* d_index,
                                   DataT* d_out0,
                                   DataT* d_out1,
                                   int batch_size,
                                   int n_obs,
                                   cudaStream_t stream)
{
  if (n_obs == 1) {
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int i) {
        (d_mask[i] ? d_out1 : d_out0)[d_index[i]] = d_in[i];
      });
  } else {
    int TPB = std::min(64, n_obs);
    divide_by_mask_kernel<<<batch_size, TPB, 0, stream>>>(
      d_in, d_mask, d_index, d_out0, d_out1, n_obs);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

/* A structure that defines a function to get the column of an element of
 * a matrix from its index. This makes possible a 2d scan with thrust.
 * Found in thrust/examples/scan_matrix_by_rows.cu
 */
struct which_col {
  MLCommon::FastIntDiv divisor;
  __host__ which_col(int col_length) : divisor(col_length) {}
  __host__ __device__ int operator()(int idx) const { return idx / divisor; }
};

/**
 * Batch division by minimum value step 1: build an index of which sub-batch
 * each series belongs to, an index of the position of each series in its new
 * batch, and measure the size of each sub-batch
 *
 * @param[in]  d_matrix   Matrix of the values to minimize
 *                        Shape: (batch_size, n_sub)
 * @param[out] d_batch    Which sub-batch each series belongs to
 * @param[out] d_index    Index of each series in its new batch
 * @param[out] h_size     Size of each sub-batch (host)
 * @param[in]  batch_size Batch size
 * @param[in]  n_sub      Number of sub-batches
 * @param[in]  stream     CUDA stream
 */
template <typename DataT>
inline void divide_by_min_build_index(const DataT* d_matrix,
                                      int* d_batch,
                                      int* d_index,
                                      int* h_size,
                                      int batch_size,
                                      int n_sub,
                                      cudaStream_t stream)
{
  auto counting = thrust::make_counting_iterator(0);

  // In the first pass, compute d_batch and initialize the matrix that will
  // be used to compute d_size and d_index (1 for the first occurrence of the
  // minimum of each row, else 0)
  rmm::device_uvector<int> cumul(batch_size * n_sub, stream);
  int* d_cumul = cumul.data();
  RAFT_CUDA_TRY(cudaMemsetAsync(d_cumul, 0, batch_size * n_sub * sizeof(int), stream));
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int i) {
      int min_id      = 0;
      DataT min_value = d_matrix[i];
      for (int j = 1; j < n_sub; j++) {
        DataT Mij = d_matrix[j * batch_size + i];
        min_id    = (Mij < min_value) ? j : min_id;
        min_value = min(Mij, min_value);
      }
      d_batch[i]                       = min_id;
      d_cumul[min_id * batch_size + i] = 1;
    });

  // In the second pass, we compute the cumulative sum of each column of this
  // mask matrix
  thrust::transform_iterator<which_col, thrust::counting_iterator<int>> t_first(
    counting, which_col(batch_size));
  thrust::inclusive_scan_by_key(
    thrust::cuda::par.on(stream), t_first, t_first + batch_size * n_sub, d_cumul, d_cumul);

  // In the third pass, we compute d_index from d_cumul and d_batch
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int i) {
      d_index[i] = d_cumul[d_batch[i] * batch_size + i] - 1;
    });

  // Finally we also compute h_size from d_cumul
  rmm::device_uvector<int> size_buffer(n_sub, stream);
  int* d_size = size_buffer.data();
  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + n_sub, [=] __device__(int j) {
    d_size[j] = d_cumul[(j + 1) * batch_size - 1];
  });
  raft::update_host(h_size, d_size, n_sub, stream);
  raft::interruptible::synchronize(stream);
}

/**
 * Batch division by minimum value step 2: create all the sub-batches
 *
 * @param[in]  d_in       Input batch
 * @param[in]  d_batch    Which sub-batch each series belongs to
 * @param[in]  d_index    Index of each series in its new sub-batch
 * @param[out] d_out      Array of pointers to the arrays of each sub-batch
 * @param[in]  n_obs      Number of data points per series
 */
template <typename DataT>
CUML_KERNEL void divide_by_min_kernel(
  const DataT* d_in, const int* d_batch, const int* d_index, DataT** d_out, int n_obs)
{
  const DataT* b_in = d_in + n_obs * blockIdx.x;
  DataT* b_out      = d_out[d_batch[blockIdx.x]] + n_obs * d_index[blockIdx.x];

  for (int i = threadIdx.x; i < n_obs; i += blockDim.x) {
    b_out[i] = b_in[i];
  }
}

/**
 * Batch division by minimum value step 2: create all the sub-batches
 *
 * @param[in]  d_in       Input batch. Each series is a contiguous chunk
 * @param[in]  d_batch    Which sub-batch each series belongs to
 * @param[in]  d_index    Index of each series in its new sub-batch
 * @param[out] hd_out     Host array of pointers to device arrays of each
 *                        sub-batch
 * @param[in]  batch_size Batch size
 * @param[in]  n_sub      Number of sub-batches
 * @param[in]  n_obs      Number of data points per series
 * @param[in]  stream     CUDA stream
 */
template <typename DataT>
inline void divide_by_min_execute(const DataT* d_in,
                                  const int* d_batch,
                                  const int* d_index,
                                  DataT** hd_out,
                                  int batch_size,
                                  int n_sub,
                                  int n_obs,
                                  cudaStream_t stream)
{
  // Create a device array of pointers to each sub-batch
  rmm::device_uvector<DataT*> out_buffer(n_sub, stream);
  DataT** d_out = out_buffer.data();
  raft::update_device(d_out, hd_out, n_sub, stream);

  if (n_obs == 1) {
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int i) {
        d_out[d_batch[i]][d_index[i]] = d_in[i];
      });
  } else {
    int TPB = std::min(64, n_obs);
    divide_by_min_kernel<<<batch_size, TPB, 0, stream>>>(d_in, d_batch, d_index, d_out, n_obs);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

/**
 * Kernel to build the division map
 *
 * @param[in]  d_id          Array of pointers to arrays containing the indices
 *                           of the members of each sub-batch
 * @param[in]  d_size        Array containing the size of each sub-batch
 * @param[out] d_id_to_pos   Array containing the position of each member in
 *                           its new sub-batch
 * @param[out] d_id_to_model Array associating each member with its
 *                           sub-batch
 */
CUML_KERNEL void build_division_map_kernel(const int* const* d_id,
                                           const int* d_size,
                                           int* d_id_to_pos,
                                           int* d_id_to_model)
{
  const int* b_id = d_id[blockIdx.x];
  int b_size      = d_size[blockIdx.x];

  for (int i = threadIdx.x; i < b_size; i += blockDim.x) {
    int original_id            = b_id[i];
    d_id_to_pos[original_id]   = i;
    d_id_to_model[original_id] = blockIdx.x;
  }
}

/**
 * Build a map to associate each batch member with a model and index in the
 * associated sub-batch
 *
 * @param[in]  hd_id         Host array of pointers to device arrays containing
 *                           the indices of the members of each sub-batch
 * @param[in]  h_size        Host array containing the size of each sub-batch
 * @param[out] d_id_to_pos   Device array containing the position of each
 *                           member in its new sub-batch
 * @param[out] d_id_to_model Device array associating each member with its
 *                           sub-batch
 * @param[in]  batch_size    Batch size
 * @param[in]  n_sub         Number of sub-batches
 * @param[in]  stream        CUDA stream
 */
inline void build_division_map(const int* const* hd_id,
                               const int* h_size,
                               int* d_id_to_pos,
                               int* d_id_to_model,
                               int batch_size,
                               int n_sub,
                               cudaStream_t stream)
{
  // Copy the pointers to the id trackers of each sub-batch to the device
  rmm::device_uvector<int*> id_ptr_buffer(n_sub, stream);
  const int** d_id = const_cast<const int**>(id_ptr_buffer.data());
  raft::update_device(d_id, hd_id, n_sub, stream);

  // Copy the size of each sub-batch to the device
  rmm::device_uvector<int> size_buffer(n_sub, stream);
  int* d_size = size_buffer.data();
  raft::update_device(d_size, h_size, n_sub, stream);

  int avg_size = batch_size / n_sub;
  int TPB      = avg_size > 256 ? 256 : (avg_size > 128 ? 128 : (avg_size > 64 ? 64 : 32));
  build_division_map_kernel<<<n_sub, TPB, 0, stream>>>(d_id, d_size, d_id_to_pos, d_id_to_model);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Kernel to merge the series into a single batch
 *
 * @param[in]  d_in        Array of pointers to arrays containing the
 *                         sub-batches
 * @param[in]  d_id_to_pos Array containing the position of each member in its
 *                         new sub-batch
 * @param[in]  d_id_to_sub Array associating each member with its sub-batch
 * @param[out] d_out       Output merged batch
 * @param[in]  n_obs       Number of observations (or forecasts) per series
 */
template <typename DataT>
CUML_KERNEL void merge_series_kernel(
  const DataT* const* d_in, const int* d_id_to_pos, const int* d_id_to_sub, DataT* d_out, int n_obs)
{
  const DataT* b_in = d_in[d_id_to_sub[blockIdx.x]] + n_obs * d_id_to_pos[blockIdx.x];
  DataT* b_out      = d_out + n_obs * blockIdx.x;

  for (int i = threadIdx.x; i < n_obs; i += blockDim.x) {
    b_out[i] = b_in[i];
  }
}

/**
 * Merge multiple sub-batches into one batch according to the maps that
 * associate each id in the unique batch to a sub-batch and a position in
 * this sub-batch.
 *
 * @param[in]  hd_in       Host array of pointers to device arrays containing
 *                         the sub-batches
 * @param[in]  d_id_to_pos Device array containing the position of each member
 *                         in its new sub-batch
 * @param[in]  d_id_to_sub Device array associating each member with its
 *                         sub-batch
 * @param[out] d_out       Output merged batch
 * @param[in]  batch_size  Batch size
 * @param[in]  n_sub       Number of sub-batches
 * @param[in]  n_obs       Number of observations (or forecasts) per series
 * @param[in]  stream      CUDA stream
 */
template <typename DataT>
inline void merge_series(const DataT* const* hd_in,
                         const int* d_id_to_pos,
                         const int* d_id_to_sub,
                         DataT* d_out,
                         int batch_size,
                         int n_sub,
                         int n_obs,
                         cudaStream_t stream)
{
  // Copy the pointers to each sub-batch to the device
  rmm::device_uvector<DataT*> in_buffer(n_sub, stream);
  const DataT** d_in = const_cast<const DataT**>(in_buffer.data());
  raft::update_device(d_in, hd_in, n_sub, stream);

  int TPB = std::min(64, n_obs);
  merge_series_kernel<<<batch_size, TPB, 0, stream>>>(d_in, d_id_to_pos, d_id_to_sub, d_out, n_obs);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace TimeSeries
}  // namespace ML
