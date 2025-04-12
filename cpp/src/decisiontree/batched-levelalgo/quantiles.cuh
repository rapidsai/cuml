/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "quantiles.h"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/fill.h>
#include <thrust/unique.h>

#include <iostream>
#include <memory>

namespace ML {
namespace DT {

template <typename T>
static __global__ void computeQuantilesKernel(
  T* quantiles, int* n_bins, const T* sorted_data, const int max_n_bins, const int n_rows)
{
  double bin_width = static_cast<double>(n_rows) / max_n_bins;

  for (int bin = threadIdx.x; bin < max_n_bins; bin += blockDim.x) {
    // get index by interpolation
    int idx        = int(round((bin + 1) * bin_width)) - 1;
    idx            = min(max(0, idx), n_rows - 1);
    quantiles[bin] = sorted_data[idx];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    // make quantiles unique, in-place
    // thrust::seq to explicitly disable cuda dynamic parallelism here
    auto new_last = thrust::unique(thrust::seq, quantiles, quantiles + max_n_bins);
    // get the unique count
    *n_bins = new_last - quantiles;
  }

  __syncthreads();
  return;
}

template <typename T>
using QuantileReturnValue = std::tuple<ML::DT::Quantiles<T, int>,
                                       std::shared_ptr<rmm::device_uvector<T>>,
                                       std::shared_ptr<rmm::device_uvector<int>>>;

template <typename T>
QuantileReturnValue<T> computeQuantiles(
  const raft::handle_t& handle, const T* data, int max_n_bins, int n_rows, int n_cols)
{
  raft::common::nvtx::push_range("computeQuantiles");
  auto stream               = handle.get_stream();
  size_t temp_storage_bytes = 0;  // for device radix sort
  rmm::device_uvector<T> sorted_column(n_rows, stream);
  // acquire device vectors to store the quantiles + offsets
  auto quantiles_array = std::make_shared<rmm::device_uvector<T>>(n_cols * max_n_bins, stream);
  auto n_bins_array    = std::make_shared<rmm::device_uvector<int>>(n_cols, stream);

  // get temp_storage_bytes for sorting
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortKeys(
    nullptr, temp_storage_bytes, data, sorted_column.data(), n_rows, 0, 8 * sizeof(T), stream));
  // allocate total memory needed for parallelized sorting
  rmm::device_uvector<char> d_temp_storage(temp_storage_bytes, stream);
  for (int col = 0; col < n_cols; col++) {
    raft::common::nvtx::push_range("sorting columns");
    int col_offset = col * n_rows;
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortKeys((void*)(d_temp_storage.data()),
                                                 temp_storage_bytes,
                                                 data + col_offset,
                                                 sorted_column.data(),
                                                 n_rows,
                                                 0,
                                                 8 * sizeof(T),
                                                 stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    raft::common::nvtx::pop_range();  // sorting columns

    int n_blocks        = 1;
    int n_threads       = min(1024, max_n_bins);
    int quantile_offset = col * max_n_bins;
    int bins_offset     = col;
    raft::common::nvtx::push_range("computeQuantilesKernel @quantile.cuh");
    computeQuantilesKernel<<<n_blocks, n_threads, 0, stream>>>(
      quantiles_array->data() + quantile_offset,
      n_bins_array->data() + bins_offset,
      sorted_column.data(),
      max_n_bins,
      n_rows);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
    RAFT_CUDA_TRY(cudaGetLastError());
    raft::common::nvtx::pop_range();  // computeQuatilesKernel
  }
  // encapsulate the device pointers under a Quantiles struct
  Quantiles<T, int> quantiles;
  quantiles.quantiles_array = quantiles_array->data();
  quantiles.n_bins_array    = n_bins_array->data();
  raft::common::nvtx::pop_range();  // computeQuantiles
  return std::make_tuple(quantiles, quantiles_array, n_bins_array);
}

}  // namespace DT
}  // namespace ML
