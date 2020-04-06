/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <cub/device/device_scan.cuh>

#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "cuml/common/cuml_allocator.hpp"
#include "utils.h"

/// TODO: unit tests!

namespace ML {
namespace TimeSeries {

/**
 * @todo: docs
 */
void cumulative_sum_helper(const bool* mask, int* cumul, int batch_size,
                           std::shared_ptr<deviceAllocator> allocator,
                           cudaStream_t stream) {
  // Determine temporary storage size
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, mask, cumul,
                                batch_size, stream);

  // Allocate temporary storage
  void* d_temp_storage = allocator->allocate(temp_storage_bytes, stream);

  // Execute the scan
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mask, cumul,
                                batch_size, stream);

  // Deallocate temporary storage
  allocator->deallocate(d_temp_storage, temp_storage_bytes, stream);
}

/**
 * Batch division by mask step 1: build an index of the position of each series
 * in its new batch and measure the size of each sub-batch
 *
 * @tparam     DataT      Data type
 * @param[in]  d_mask     Boolean mask
 * @param[out] d_index    Index of each series in its new batch
 * @param[in]  batch_size Batch size
 * @param[in]  allocator  Device memory allocator
 * @param[in]  stream     CUDA stream
 * @return The number of 'true' series in the mask
 */
inline int divide_by_mask_build_index(
  const bool* d_mask, int* d_index, int batch_size,
  std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream) {
  // Inverse mask
  MLCommon::device_buffer<bool> inv_mask(allocator, stream, batch_size);
  thrust::transform(thrust::cuda::par.on(stream), d_mask, d_mask + batch_size,
                    inv_mask.data(), thrust::logical_not<bool>());

  // Cumulative sum of the inverse mask
  MLCommon::device_buffer<int> index0(allocator, stream, batch_size);
  cumulative_sum_helper(inv_mask.data(), index0.data(), batch_size, allocator,
                        stream);

  // Cumulative sum of the mask
  MLCommon::device_buffer<int> index1(allocator, stream, batch_size);
  cumulative_sum_helper(d_mask, index1.data(), batch_size, allocator, stream);

  // Combine both cumulative sums according to the mask and subtract 1
  const int* d_index0 = index0.data();
  const int* d_index1 = index1.data();
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int i) {
                     d_index[i] = (d_mask[i] ? d_index1[i] : d_index0[i]) - 1;
                   });

  // Compute and return the number of true elements in the mask
  int true_elements;
  MLCommon::updateHost(&true_elements, index1.data() + batch_size - 1, 1,
                       stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return true_elements;
}

/**
 * @todo: docs
 *        version of the kernel for small series (e.g the index: 1 element)
 */
template <typename DataT>
__global__ void divide_by_mask_kernel(const DataT* d_in, const bool* d_mask,
                                      const int* d_index, DataT* d_out0,
                                      DataT* d_out1, int n_obs) {
  const DataT* b_in = d_in + n_obs * blockIdx.x;
  DataT* b_out =
    (d_mask[blockIdx.x] ? d_out1 : d_out0) + n_obs * d_index[blockIdx.x];

  for (int i = threadIdx.x; i < n_obs; i += blockDim.x) {
    b_out[i] = b_in[i];
  }
}

/**
 * Batch division by mask step 2: create both sub-batches from the mask and
 * index
 *
 * @tparam     DataT      Data type
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
inline void divide_by_mask_execute(const DataT* d_in, const bool* d_mask,
                                   const int* d_index, DataT* d_out0,
                                   DataT* d_out1, int batch_size, int n_obs,
                                   cudaStream_t stream) {
  int TPB = std::min(64, n_obs);
  divide_by_mask_kernel<<<batch_size, TPB, 0, stream>>>(d_in, d_mask, d_index,
                                                        d_out0, d_out1, n_obs);
  CUDA_CHECK(cudaPeekAtLastError());
}

/* A structure that defines a function to get the column of an element of
 * a matrix from its index. This makes possible a 2d scan with thrust.
 * Found in thrust/examples/scan_matrix_by_rows.cu
 */
struct which_col : thrust::unary_function<int, int> {
  int col_length;
  __host__ __device__ which_col(int col_length_) : col_length(col_length_) {}
  __host__ __device__ int operator()(int idx) const { return idx / col_length; }
};

/**
 * Batch division by minimum value step 1: build an index of which sub-batch
 * each series belongs to, an index of the position of each series in its new
 * batch, and measure the size of each sub-batch
 *
 * @tparam     DataT      Data type
 * @param[in]  d_matrix   Matrix of the values to minimize
 *                        Shape: (batch_size, n_sub)
 * @param[out] d_batch    Which sub-batch each series belongs to
 * @param[out] d_index    Index of each series in its new batch
 * @param[out] h_size     Size of each sub-batch (host)
 * @param[in]  batch_size Batch size
 * @param[in]  n_sub      Number of sub-batches
 * @param[in]  allocator  Device memory allocator
 * @param[in]  stream     CUDA stream
 */
template <typename DataT>
inline void divide_by_min_build_index(
  const DataT* d_matrix, int* d_batch, int* d_index, int* h_size,
  int batch_size, int n_sub, std::shared_ptr<deviceAllocator> allocator,
  cudaStream_t stream) {
  auto counting = thrust::make_counting_iterator(0);

  // In the first pass, compute d_batch and initialize the matrix that will
  // be used to compute d_size and d_index (1 for the first occurence of the
  // minimum of each row, else 0)
  MLCommon::device_buffer<int> cumul(allocator, stream, batch_size * n_sub);
  int* d_cumul = cumul.data();
  CUDA_CHECK(
    cudaMemsetAsync(d_cumul, 0, batch_size * n_sub * sizeof(int), stream));
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int i) {
                     int min_id = 0;
                     DataT min_value = d_matrix[i];
                     for (int j = 1; j < n_sub; j++) {
                       DataT Mij = d_matrix[j * batch_size + i];
                       min_id = (Mij < min_value) ? j : min_id;
                       min_value = min(Mij, min_value);
                     }
                     d_batch[i] = min_id;
                     d_cumul[min_id * batch_size + i] = 1;
                   });

  // In the second pass, we compute the cumulative sum of each column of this
  // mask matrix
  thrust::transform_iterator<which_col, thrust::counting_iterator<int>> t_first(
    counting, which_col(batch_size));
  thrust::inclusive_scan_by_key(thrust::cuda::par.on(stream), t_first,
                                t_first + batch_size * n_sub, d_cumul, d_cumul);

  // In the third pass, we compute d_index from d_cumul and d_batch
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int i) {
                     d_index[i] = d_cumul[d_batch[i] * batch_size + i] - 1;
                   });

  // Finally we also compute h_size from d_cumul
  MLCommon::device_buffer<int> size_buffer(allocator, stream, n_sub);
  int* d_size = size_buffer.data();
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + n_sub,
    [=] __device__(int j) { d_size[j] = d_cumul[(j + 1) * batch_size - 1]; });
  MLCommon::updateHost(h_size, d_size, n_sub, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @todo: docs
 *        version of the kernel for small series (e.g the index: 1 element)
 */
template <typename DataT>
__global__ void divide_by_min_kernel(const DataT* d_in, const int* d_batch,
                                     const int* d_index, DataT** d_out,
                                     int n_obs) {
  const DataT* b_in = d_in + n_obs * blockIdx.x;
  DataT* b_out = d_out[d_batch[blockIdx.x]] + n_obs * d_index[blockIdx.x];

  for (int i = threadIdx.x; i < n_obs; i += blockDim.x) {
    b_out[i] = b_in[i];
  }
}

/**
 * Batch division by minimum value step 2: create all the sub-batches
 *
 * @tparam     DataT      Data type
 * @param[in]  d_in       Input batch. Each series is a contiguous chunk
 * @param[in]  d_batch    Which sub-batch each series belongs to
 * @param[in]  d_index    Index of each series in its new sub-batch
 * @param[out] hd_out     Host array of pointers to device arrays of each
 *                        sub-batch
 * @param[in]  batch_size Batch size
 * @param[in]  n_sub      Number of sub-batches
 * @param[in]  n_obs      Number of data points per series
 * @param[in]  allocator  Device memory allocator
 * @param[in]  stream     CUDA stream
 */
template <typename DataT>
inline void divide_by_min_execute(const DataT* d_in, const int* d_batch,
                                  const int* d_index, DataT** hd_out,
                                  int batch_size, int n_sub, int n_obs,
                                  std::shared_ptr<deviceAllocator> allocator,
                                  cudaStream_t stream) {
  // Create a device array of pointers to each sub-batch
  MLCommon::device_buffer<DataT*> out_buffer(allocator, stream, n_sub);
  DataT** d_out = out_buffer.data();
  MLCommon::updateDevice(d_out, hd_out, n_sub, stream);

  int TPB = std::min(64, n_obs);
  divide_by_min_kernel<<<batch_size, TPB, 0, stream>>>(d_in, d_batch, d_index,
                                                       d_out, n_obs);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @todo: docs
 */
__global__ void build_division_map_kernel(const int* const* d_id,
                                          const int* d_size, int* d_id_to_pos,
                                          int* d_id_to_model) {
  const int* b_id = d_id[blockIdx.x];
  int b_size = d_size[blockIdx.x];

  for (int i = threadIdx.x; i < b_size; i += blockDim.x) {
    int original_id = b_id[i];
    d_id_to_pos[original_id] = i;
    d_id_to_model[original_id] = blockIdx.x;
  }
}

/**
 * @todo: docs
 */
inline void build_division_map(const int* const* hd_id, const int* h_size,
                               int* d_id_to_pos, int* d_id_to_model,
                               int batch_size, int n_sub,
                               std::shared_ptr<deviceAllocator> allocator,
                               cudaStream_t stream) {
  // Copy the pointers to the id trackers of each sub-batch to the device
  MLCommon::device_buffer<int*> id_ptr_buffer(allocator, stream, n_sub);
  const int** d_id = const_cast<const int**>(id_ptr_buffer.data());
  MLCommon::updateDevice(d_id, hd_id, n_sub, stream);

  // Copy the size of each sub-batch to the device
  MLCommon::device_buffer<int> size_buffer(allocator, stream, n_sub);
  int* d_size = size_buffer.data();
  MLCommon::updateDevice(d_size, h_size, n_sub, stream);

  int avg_size = batch_size / n_sub;
  int TPB = avg_size > 128 ? 256 : (avg_size > 64 ? 128 : 64);
  build_division_map_kernel<<<n_sub, TPB, 0, stream>>>(
    d_id, d_size, d_id_to_pos, d_id_to_model);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace TimeSeries
}  // namespace ML
