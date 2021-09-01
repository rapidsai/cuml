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
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>
#include "quantile.h"

#include <common/nvtx.hpp>

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
void computeQuantiles(T* quantiles,
                      int n_bins,
                      const T* data,
                      int n_rows,
                      int n_cols,
                      const std::shared_ptr<raft::mr::device::allocator> device_allocator,
                      cudaStream_t stream)
{
  thrust::fill(
    thrust::cuda::par(*device_allocator).on(stream), quantiles, quantiles + n_bins * n_cols, 0.0);
  // Determine temporary device storage requirements
  std::unique_ptr<device_buffer<char>> d_temp_storage = nullptr;
  size_t temp_storage_bytes                           = 0;

  std::unique_ptr<device_buffer<T>> single_column_sorted = nullptr;
  single_column_sorted = std::make_unique<device_buffer<T>>(device_allocator, stream, n_rows);

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            data,
                                            single_column_sorted->data(),
                                            n_rows,
                                            0,
                                            8 * sizeof(T),
                                            stream));

  // Allocate temporary storage for sorting
  d_temp_storage =
    std::make_unique<device_buffer<char>>(device_allocator, stream, temp_storage_bytes);

  // Compute quantiles column by column
  for (int col = 0; col < n_cols; col++) {
    int col_offset      = col * n_rows;
    int quantile_offset = col * n_bins;

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys((void*)d_temp_storage->data(),
                                              temp_storage_bytes,
                                              data + col_offset,
                                              single_column_sorted->data(),
                                              n_rows,
                                              0,
                                              8 * sizeof(T),
                                              stream));

    int blocks = raft::ceildiv(n_bins, 128);

    computeQuantilesSorted<<<blocks, 128, 0, stream>>>(
      quantiles + quantile_offset, n_bins, single_column_sorted->data(), n_rows);

    CUDA_CHECK(cudaGetLastError());
  }

  return;
}

}  // namespace DT
}  // namespace ML
