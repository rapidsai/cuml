/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/fill.h>

#include <raft/common/nvtx.hpp>

#include "quantiles.h"

namespace ML {
namespace DT {

template <typename T>
__global__ void computeQuantilesBatchSorted(
  T* quantiles, int* useful_nbins, const T* sorted_data, const int n_bins, const int length);

template <typename T>
void computeQuantiles(Quantiles<T, int>& quantiles,
                      int n_bins,
                      const T* data,
                      int n_rows,
                      int n_cols,
                      int n_streams,
                      const raft::handle_t& handle)
{
  raft::common::nvtx::push_range("computeQuantiles");
  // auto quantiles_array = std::make_shared<rmm::device_uvector<T>>(n_bins * n_cols,
  // handle.get_stream()); auto n_uniquebins_array =
  // std::make_shared<rmm::device_uvector<int>>(n_cols, handle.get_stream());

  int prllsm = n_streams;  // the parallism to be used stream-wise and omp-thread-wise
  size_t temp_storage_bytes = 0;
  rmm::device_uvector<T> all_column_sorted(n_cols * n_rows, handle.get_stream());

  raft::common::nvtx::push_range("sorting columns");
  // get temp_storage_bytes for sorting
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            data,
                                            all_column_sorted.data(),
                                            n_rows,
                                            0,
                                            8 * sizeof(T),
                                            handle.get_stream()));
  // allocate total memory needed for parallelized sorting
  rmm::device_uvector<char> d_temp_storage(prllsm * temp_storage_bytes, handle.get_stream());
  // handle for sorting across multiple streams
  // auto sorting_handle =
  //   raft::handle_t(rmm::cuda_stream_per_thread, std::make_shared<rmm::cuda_stream_pool>(prllsm));
#pragma omp parallel for num_threads(prllsm)
  for (int parcol = 0; parcol < n_cols; parcol++) {
    int thread_id  = omp_get_thread_num();
    auto s         = handle.get_stream_from_stream_pool(thread_id);
    int col_offset = parcol * n_rows;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      (void*)(d_temp_storage.data() + temp_storage_bytes * thread_id),
      temp_storage_bytes,
      data + col_offset,
      all_column_sorted.data() + col_offset,
      n_rows,
      0,
      8 * sizeof(T),
      s));
    s.synchronize();
  }
  handle.sync_stream_pool();
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  raft::common::nvtx::pop_range();  // sorting columns

  // do the quantile computation parallelizing across cols too across CTAs
  int blocks      = n_cols;
  size_t smemsize = n_bins * sizeof(T);
  raft::common::nvtx::push_range("computeQuantilesBatchSorted @quantile.cuh");
  computeQuantilesBatchSorted<<<blocks, 128, smemsize, handle.get_stream()>>>(
    quantiles.quantiles_array,
    quantiles.n_uniquebins_array,
    all_column_sorted.data(),
    n_bins,
    n_rows);
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  CUDA_CHECK(cudaGetLastError());
  raft::common::nvtx::pop_range();  // computeQuatilesBatchSorted
  raft::common::nvtx::pop_range();  // computeQuantiles
  // Quantiles<T, int> q = {quantiles_array, n_uniquebins_array};
  // return q;
  // return {quantiles_array->data(), n_uniquebins_array->data()};
}

}  // namespace DT
}  // namespace ML
