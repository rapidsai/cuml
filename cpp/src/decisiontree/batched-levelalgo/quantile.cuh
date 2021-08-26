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
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <cub/cub.cuh>
#include <cuml/common/device_buffer.hpp>
#include <memory>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>

namespace ML {
namespace DT {

using device_allocator = raft::mr::device::allocator;
template <typename T>
using device_buffer = raft::mr::device::buffer<T>;

template <typename T>
__global__ void computeQuantilesSorted(T* quantiles,
                                       const int n_bins,
                                       const T* sorted_data,
                                       const int length)
{
  int tid          = threadIdx.x + blockIdx.x * blockDim.x;
  double bin_width = static_cast<double>(length) / n_bins;
  int index        = int(round((tid + 1) * bin_width)) - 1;
  index            = min(max(0, index), length - 1);
  if (tid < n_bins) { quantiles[tid] = sorted_data[index]; }

  return;
}

template <typename T>
std::shared_ptr<MLCommon::device_buffer<T>> computeQuantiles(
  int n_bins, const T* data, int n_rows, int n_cols, const raft::handle_t& handle)
{
  auto quantiles = std::make_shared<MLCommon::device_buffer<T>>(
    handle.get_device_allocator(), handle.get_stream(), n_bins * n_cols);
  thrust::fill(thrust::cuda::par(*handle.get_device_allocator()).on(handle.get_stream()),
               quantiles->begin(),
               quantiles->begin() + n_bins * n_cols,
               0.0);
  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;

  MLCommon::device_buffer<T> single_column_sorted(
    handle.get_device_allocator(), handle.get_stream(), n_rows);

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            data,
                                            single_column_sorted.data(),
                                            n_rows,
                                            0,
                                            8 * sizeof(T),
                                            handle.get_stream()));

  // Allocate temporary storage for sorting
  MLCommon::device_buffer<char> d_temp_storage(
    handle.get_device_allocator(), handle.get_stream(), temp_storage_bytes);

  // Compute quantiles column by column
  for (int col = 0; col < n_cols; col++) {
    int col_offset      = col * n_rows;
    int quantile_offset = col * n_bins;

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys((void*)d_temp_storage.data(),
                                              temp_storage_bytes,
                                              data + col_offset,
                                              single_column_sorted.data(),
                                              n_rows,
                                              0,
                                              8 * sizeof(T),
                                              handle.get_stream()));

    int blocks = raft::ceildiv(n_bins, 128);

    auto s = handle.get_stream();
    computeQuantilesSorted<<<blocks, 128, 0, s>>>(
      quantiles->data() + quantile_offset, n_bins, single_column_sorted.data(), n_rows);

    CUDA_CHECK(cudaGetLastError());
  }

  return quantiles;
}

}  // namespace DT
}  // namespace ML
