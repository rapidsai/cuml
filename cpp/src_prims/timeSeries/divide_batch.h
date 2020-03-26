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
#include <thrust/transform.h>
#include <cub/device/device_scan.cuh>

#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "cuml/common/cuml_allocator.hpp"
#include "utils.h"

namespace MLCommon {
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
 * Batch division step 1: build an index of the position of each series
 * in its new batch and measure the size of each sub-batch
 *
 * @tparam     DataT      Data type
 * @param[in]  d_mask     Boolean mask
 * @param[out] d_index    Index of each series in its new batch
 * @param[in]  batch_size Batch size
 * @param[in]  n_obs      Number of data points per series
 * @param[in]  allocator  Device memory allocator
 * @param[in]  stream     CUDA stream
 * @return The number of 'true' series in the mask
 */
inline int divide_batch_build_index(const bool* d_mask, int* d_index,
                                    int batch_size, int n_obs,
                                    std::shared_ptr<deviceAllocator> allocator,
                                    cudaStream_t stream) {
  // Inverse mask
  device_buffer<bool> inv_mask(allocator, stream, batch_size);
  thrust::transform(thrust::cuda::par.on(stream), d_mask, d_mask + batch_size,
                    inv_mask.data(), thrust::logical_not<bool>());

  // Cumulative sum of the inverse mask
  device_buffer<int> index0(allocator, stream, batch_size);
  cumulative_sum_helper(inv_mask.data(), index0.data(), batch_size, allocator,
                        stream);

  // Cumulative sum of the mask
  device_buffer<int> index1(allocator, stream, batch_size);
  cumulative_sum_helper(d_mask, index1.data(), batch_size, allocator, stream);

  myPrintDevVector("idx0", index0.data(), batch_size);
  myPrintDevVector("idx1", index1.data(), batch_size);

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
  updateHost(&true_elements, index1.data() + batch_size - 1, 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return true_elements;
}

/**
 * @todo: docs
 */
template <typename DataT>
__global__ void divide_batch_kernel(const DataT* d_in, const bool* d_mask,
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
 * Batch division step 2: create both sub-batches from the mask and index
 *
 * @tparam     DataT      Data type
 * @param[in]  d_in       Input batch. Each series is a contiguous chunk
 * @param[in]  d_mask     Boolean mask
 * @param[in]  d_index    Index of each series in its new batch
 * @param[out] d_out0     The sub-batch for the 'false' members
 * @param[out] d_out1     The sub-batch for the 'true' members
 * @param[in]  batch_size Batch size
 * @param[in]  n_obs      Number of data points per series
 * @param[in]  allocator  Device memory allocator
 * @param[in]  stream     CUDA stream
 */
template <typename DataT>
void divide_batch_execute(const DataT* d_in, const bool* d_mask,
                          const int* d_index, DataT* d_out0, DataT* d_out1,
                          int batch_size, int n_obs, cudaStream_t stream) {
  int TPB = std::min(64, n_obs);
  divide_batch_kernel<<<batch_size, TPB, 0, stream>>>(d_in, d_mask, d_index,
                                                      d_out0, d_out1, n_obs);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace TimeSeries
}  // namespace MLCommon
