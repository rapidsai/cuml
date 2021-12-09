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

#include <thrust/fill.h>
#include <cub/cub.cuh>
#include <memory>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <thrust/device_vector.h>
#include <rmm/exec_policy.hpp>
#include <iostream>
#include <fstream>

#include <common/nvtx.hpp>

namespace ML {
namespace DT {

template <typename T>
__global__ void computeQuantilesSorted(T* quantiles,
                                       const int n_bins,
                                       const T* sorted_data,
                                       const int length);

template <typename T>
  auto computeQuantiles(
  int n_bins, const T* data, int n_rows, int n_cols, const raft::handle_t& handle)
{
  auto quantiles = std::make_shared<rmm::device_uvector<T>>(n_bins * n_cols, handle.get_stream());
  auto q_offsets = std::make_shared<rmm::device_uvector<int>>(n_cols, handle.get_stream());
  thrust::fill(rmm::exec_policy(handle.get_stream()),
               quantiles->begin(),
               quantiles->begin() + n_bins * n_cols,
               0.0);
  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;

  rmm::device_uvector<T> single_column_sorted(n_rows, handle.get_stream());

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            data,
                                            single_column_sorted.data(),
                                            n_rows,
                                            0,
                                            8 * sizeof(T),
                                            handle.get_stream()));

  // Allocate temporary storage for sorting
  rmm::device_uvector<char> d_temp_storage(temp_storage_bytes, handle.get_stream());

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

  // print the unique quantiles and store it in file
  std::vector<int> h_q_offsets(n_cols, 0);
  thrust::device_vector<T> d_q(n_bins, 0);
  auto compacted_quantiles = std::make_shared<rmm::device_uvector<T>>(0, handle.get_stream());
  for (int col=0; col < n_cols; ++col) {
    auto first = quantiles->begin() + n_bins * col;
    thrust::copy(thrust::device, first, first + n_bins, d_q.begin());
    auto new_last = thrust::unique(thrust::device, d_q.begin(), d_q.begin()+n_bins);
    int n_uniques = new_last - d_q.begin();
    h_q_offsets[col] = h_q_offsets[col? col-1 : 0] + n_uniques;
    int old_size = compacted_quantiles->size();
    compacted_quantiles->resize(old_size + n_uniques, handle.get_stream());
    thrust::copy(thrust::device, d_q.begin(), new_last, compacted_quantiles->begin() + old_size);
    }
  raft::update_device(q_offsets->data(), h_q_offsets.data(), n_cols, handle.get_stream());

  return std::make_pair(compacted_quantiles, q_offsets);
}

}  // namespace DT
}  // namespace ML
