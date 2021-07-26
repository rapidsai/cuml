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
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include "quantile.h"

#include <common/nvtx.hpp>

namespace ML {
namespace DT {

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
void computeQuantiles(
  T* quantiles, int n_bins, const T* data, int n_rows, int n_cols, cudaStream_t stream)
{
  thrust::fill(rmm::exec_policy(stream), quantiles, quantiles + n_bins * n_cols, 0.0);
  // Determine temporary device storage requirements
  std::unique_ptr<rmm::device_uvector<char>> d_temp_storage = nullptr;
  size_t temp_storage_bytes                                 = 0;

  std::unique_ptr<rmm::device_uvector<T>> single_column_sorted = nullptr;
  single_column_sorted = std::make_unique<rmm::device_uvector<T>>(n_rows, stream);

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            data,
                                            single_column_sorted->data(),
                                            n_rows,
                                            0,
                                            8 * sizeof(T),
                                            stream));

  // Allocate temporary storage for sorting
  d_temp_storage = std::make_unique<rmm::device_uvector<char>>(temp_storage_bytes, stream);

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
