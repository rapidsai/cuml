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

#include <omp.h>
#include <thrust/fill.h>
#include <cub/cub.cuh>
#include <memory>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <rmm/exec_policy.hpp>
#include <iostream>
#include <fstream>

#include <common/nvtx.hpp>

// #define KERNEL 0

namespace ML {
namespace DT {

template <typename T>
__global__ void batchUniqueKernel(T* quantiles, int* useful_nbins, const int n_bins){
  extern __shared__ char smem[];
  __shared__ int unq_nbins;
  auto* feature_quantiles = (T*)smem;

  for (int i = threadIdx.x; i < n_bins; i += blockDim.x){
    feature_quantiles[i] = quantiles[blockIdx.x * n_bins + i];
  }

  __syncthreads();

  if(threadIdx.x == 0) {
    auto new_last = thrust::unique(thrust::device, feature_quantiles, feature_quantiles + n_bins);
    useful_nbins[blockIdx.x] = unq_nbins = new_last - feature_quantiles;
    // printf("useful_nbins[%d]:%d\n", blockIdx.x, unq_nbins);
  }

  __syncthreads();

  for (int i = threadIdx.x; i < n_bins; i += blockDim.x) {
    if(i >= unq_nbins) break;
    quantiles[blockIdx.x * n_bins + i] = feature_quantiles[i];
  }

}

template <typename T>
__global__ void computeQuantilesBatchSorted(T* quantiles,
                                       int* useful_nbins,
                                       const T* sorted_data,
                                       const int n_bins,
                                       const int n_rows)
{
  extern __shared__ char smem[];
  auto* feature_quantiles = (T*)smem;
  __shared__ int unq_nbins;
  int col = blockIdx.x; // each col per block
  int data_base = col * n_rows;
  double bin_width = static_cast<double>(n_rows) / n_bins;

  for (int bin = threadIdx.x; bin < n_bins; bin += blockDim.x) {
    int data_offst        = int(round((bin + 1) * bin_width)) - 1;
    data_offst            = min(max(0, data_offst), n_rows - 1);
    feature_quantiles[bin] = sorted_data[data_base + data_offst];
  }

  __syncthreads();

  if(threadIdx.x == 0) {
    auto new_last = thrust::unique(thrust::device, feature_quantiles, feature_quantiles + n_bins);
    useful_nbins[blockIdx.x] = unq_nbins = new_last - feature_quantiles;
  }

  __syncthreads();

  for (int bin = threadIdx.x; bin < n_bins; bin += blockDim.x) {
    if(bin >= unq_nbins) break;
    quantiles[col * n_bins + bin] = feature_quantiles[bin];
  }

  return;
}

template <typename T>
__global__ void computeQuantilesSorted(T* quantiles,
                                       const int n_bins,
                                       const T* sorted_data,
                                       const int length);

template <typename T>
void computeQuantilesHelperV5(
  const T* data, std::shared_ptr<rmm::device_uvector<T>>& quantiles, std::shared_ptr<rmm::device_uvector<int>>& useful_nbins, int n_bins, int n_rows, int n_cols, const raft::handle_t& handle)
{
  ML::PUSH_RANGE("version 5");

  int prllsm = 2; // the parallism to be used stream-wise and omp-thread-wise
  size_t temp_storage_bytes = 0;
  rmm::device_uvector<T> all_column_sorted(n_cols * n_rows, handle.get_stream());

  ML::PUSH_RANGE("sorting columns");
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            data,
                                            all_column_sorted.data(),
                                            n_rows,
                                            0,
                                            8 * sizeof(T),
                                            handle.get_stream()));
  rmm::device_uvector<char> d_temp_storage(prllsm * temp_storage_bytes, handle.get_stream());
  auto sorting_handle = raft::handle_t(rmm::cuda_stream_per_thread, std::make_shared<rmm::cuda_stream_pool>(prllsm));
  // printf("handle.stream_pool_size:%d, sorting_handle.stream_pool_size:%d\n", (int)handle.get_stream_pool_size(), (int)sorting_handle.get_stream_pool_size());
  #pragma omp parallel for num_threads(prllsm)
  for (int parcol = 0; parcol < n_cols; parcol++) {
    // if(parcol >= n_cols) continue;
    int thread_id = omp_get_thread_num();
    // auto s        = handle.get_stream_from_stream_pool(parcol);
    auto s = sorting_handle.get_stream_from_stream_pool(thread_id);
    int col_offset      = parcol * n_rows;
    // Allocate temporary storage for sorting
    // printf("col:%d, thread_id:%d, col_offset:%d\n", parcol, thread_id, col_offset);

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys((void*)(d_temp_storage.data() + temp_storage_bytes*thread_id),
                                              temp_storage_bytes,
                                              data + col_offset,
                                              all_column_sorted.data() + col_offset,
                                              n_rows,
                                              0,
                                              8 * sizeof(T),
                                              s));
    s.synchronize();
}
  // sorting_handle.sync_stream_pool();
  // CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  ML::POP_RANGE(); // sorting columns

  // do the quantile computation parallelizing across cols too across CTAs
  int blocks = n_cols;
  size_t smemsize = n_bins * sizeof(T);
  // raft::print_device_vector("all_columns_sorted: ", all_column_sorted.begin()+n_rows, n_rows, std::cout);
  ML::PUSH_RANGE("computeQuantilesBatchSorted @quantile.cuh");
  computeQuantilesBatchSorted<<<blocks, 128, smemsize, handle.get_stream()>>>(
    quantiles->data(), useful_nbins->data(), all_column_sorted.data(), n_bins, n_rows);
  CUDA_CHECK(cudaGetLastError());
  ML::POP_RANGE(); // computeQuatilesBatchSorted
  ML::POP_RANGE(); // version 5
}

template <typename T>
void computeQuantilesHelperV4(
  const T* data, std::shared_ptr<rmm::device_uvector<T>>& quantiles, std::shared_ptr<rmm::device_uvector<int>>& useful_nbins, int n_bins, int n_rows, int n_cols, const raft::handle_t& handle)
{
  ML::PUSH_RANGE("version 4");
  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;

  rmm::device_uvector<T> all_column_sorted(n_cols * n_rows, handle.get_stream());

  std::vector<int> h_offsets(n_cols+1, 0);
  int i = 0;
  std::generate(h_offsets.begin(), h_offsets.end(), [&](){
    return n_rows * (i++);
  });
  rmm::device_uvector<int> d_offsets(n_cols+1, handle.get_stream());
  raft::update_device(d_offsets.data(), h_offsets.data(), n_cols+1, handle.get_stream());

  ML::PUSH_RANGE("sorting columns");
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            data,
                                            all_column_sorted.data(),
                                            n_rows * n_cols,
                                            n_cols,
                                            d_offsets.data(),
                                            d_offsets.data()+1,
                                            0,
                                            sizeof(T)*8,
                                            handle.get_stream()));

  // Allocate temporary storage for sorting
  // printf("temp_storage_bytes: %zu\n", temp_storage_bytes);
  rmm::device_uvector<char> d_temp_storage(temp_storage_bytes, handle.get_stream());
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys((void*)d_temp_storage.data(),
                                            temp_storage_bytes,
                                            data,
                                            all_column_sorted.data(),
                                            n_rows * n_cols,
                                            n_cols,
                                            d_offsets.data(),
                                            d_offsets.data()+1,
                                            0,
                                            sizeof(T)*8,
                                            handle.get_stream()));
  ML::POP_RANGE(); // sorting columns

  // do the quantile computation parallelizing across cols too across CTAs
  int blocks = n_cols;
  auto s = handle.get_stream();
  size_t smemsize = n_bins * sizeof(T);
  ML::PUSH_RANGE("computeQuantilesBatchSorted @quantile.cuh");
  computeQuantilesBatchSorted<<<blocks, 128, smemsize, s>>>(
    quantiles->data(), useful_nbins->data(), all_column_sorted.data(), n_bins, n_rows);
  CUDA_CHECK(cudaGetLastError());
  ML::POP_RANGE(); // computeQuatilesBatchSorted
  ML::POP_RANGE(); // version 4
}

template <typename T>
void computeQuantilesHelperV3(
  const T* data, std::shared_ptr<rmm::device_uvector<T>>& quantiles, std::shared_ptr<rmm::device_uvector<int>>& useful_nbins, int n_bins, int n_rows, int n_cols, const raft::handle_t& handle)
{
  ML::PUSH_RANGE("version 3");
  ML::PUSH_RANGE("sorting columns");
  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  rmm::device_uvector<T> all_column_sorted(n_cols * n_rows, handle.get_stream());

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            data,
                                            all_column_sorted.data(),
                                            n_rows,
                                            0,
                                            8 * sizeof(T),
                                            handle.get_stream()));

  // Allocate temporary storage for sorting
  // printf("temp_storage_bytes: %zu\n", temp_storage_bytes);
  rmm::device_uvector<char> d_temp_storage(temp_storage_bytes, handle.get_stream());
  // auto* d_temp_storage = (char*)

  // Compute quantiles column by column
  for (int col = 0; col < n_cols; col++) {
    int col_offset      = col * n_rows;

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys((void*)d_temp_storage.data(),
                                              temp_storage_bytes,
                                              data + col_offset,
                                              all_column_sorted.data() + col_offset,
                                              n_rows,
                                              0,
                                              8 * sizeof(T),
                                              handle.get_stream()));
  }
  ML::POP_RANGE(); // sorting columns
  // CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys((void*)d_temp_storage.data(),
  //                                           temp_storage_bytes,
  //                                           data,
  //                                           all_column_sorted.data(),
  //                                           n_rows * n_cols,
  //                                           n_cols,
  //                                           d_offsets.data(),
  //                                           d_offsets.data()+1,
  //                                           0,
  //                                           sizeof(T)*8,
  //                                           handle.get_stream()));

  // do the quantile computation parallelizing across cols too across CTAs
  int blocks = n_cols;
  auto s = handle.get_stream();
  size_t smemsize = n_bins * sizeof(T);
  // raft::print_device_vector("all_columns_sorted: ", all_column_sorted.begin()+n_rows, n_rows, std::cout);
  ML::PUSH_RANGE("computeQuantilesBatchSorted @quantile.cuh");
  computeQuantilesBatchSorted<<<blocks, 128, smemsize, s>>>(
    quantiles->data(), useful_nbins->data(), all_column_sorted.data(), n_bins, n_rows);
  CUDA_CHECK(cudaGetLastError());
  ML::POP_RANGE();

  // ML::PUSH_RANGE("batchUniqueKernel @quantile.cuh");
  // size_t smemSize = n_bins * sizeof(T);
  // batchUniqueKernel<<<n_cols, 128, smemSize, handle.get_stream()>>>(
  //   quantiles->data(), useful_nbins->data(), n_bins);
  // CUDA_CHECK(cudaGetLastError());
  // ML::POP_RANGE();
  // ML::PUSH_RANGE("compact computed Quantiles [host] @quantile.cuh");
  // std::vector<int> h_useful_nbins(n_cols, 0);
  // // thrust::device_vector<T> d_q(n_bins, 0);
  // // auto compacted_quantiles = std::make_shared<rmm::device_uvector<T>>(0, handle.get_stream());
  // for (int col=0; col < n_cols; ++col) {
  //   auto first = quantiles->begin() + n_bins * col;
  //   auto last = first + n_bins;
  //   // thrust::copy(thrust::device, first, first + n_bins, d_q.begin());
  //   auto new_last = thrust::unique(thrust::device, first, last);
  //   int n_uniques = new_last - first;
  //   // h_useful_nbins[col] = h_useful_nbins[col? col-1 : 0] + n_uniques;
  //   h_useful_nbins[col] = n_uniques;
  //   // int old_size = compacted_quantiles->size();
  //   // compacted_quantiles->resize(old_size + n_uniques, handle.get_stream());
  //   // thrust::copy(thrust::device, d_q.begin(), new_last, compacted_quantiles->begin() + old_size);
  //   }
  // raft::update_device(useful_nbins->data(), h_useful_nbins.data(), n_cols, handle.get_stream());
  ML::POP_RANGE(); // version 3
}

template <typename T>
void computeQuantilesHelperV2(
  const T* data, std::shared_ptr<rmm::device_uvector<T>>& quantiles, std::shared_ptr<rmm::device_uvector<int>>& useful_nbins, int n_bins, int n_rows, int n_cols, const raft::handle_t& handle)
{
  ML::PUSH_RANGE("version 2");
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

  ML::PUSH_RANGE("batchUniqueKernel @quantile.cuh");
  size_t smemSize = n_bins * sizeof(T);
  batchUniqueKernel<<<n_cols, 128, smemSize, handle.get_stream()>>>(
    quantiles->data(), useful_nbins->data(), n_bins);
  CUDA_CHECK(cudaGetLastError());
  ML::POP_RANGE(); // batchUniqueKernel: parallel thrust::unique
  ML::POP_RANGE(); // version 2
}

template <typename T>
void computeQuantilesHelperV1(
  const T* data, std::shared_ptr<rmm::device_uvector<T>>& quantiles, std::shared_ptr<rmm::device_uvector<int>>& useful_nbins, int n_bins, int n_rows, int n_cols, const raft::handle_t& handle)
{
  ML::PUSH_RANGE("version 1");
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
  ML::PUSH_RANGE("compact computed Quantiles [host] @quantile.cuh");
  std::vector<int> h_useful_nbins(n_cols, 0);
  // thrust::device_vector<T> d_q(n_bins, 0);
  // auto compacted_quantiles = std::make_shared<rmm::device_uvector<T>>(0, handle.get_stream());
  for (int col=0; col < n_cols; ++col) {
    auto first = quantiles->begin() + n_bins * col;
    auto last = first + n_bins;
    // thrust::copy(thrust::device, first, first + n_bins, d_q.begin());
    auto new_last = thrust::unique(thrust::device, first, last);
    int n_uniques = new_last - first;
    // h_useful_nbins[col] = h_useful_nbins[col? col-1 : 0] + n_uniques;
    h_useful_nbins[col] = n_uniques;
    // int old_size = compacted_quantiles->size();
    // compacted_quantiles->resize(old_size + n_uniques, handle.get_stream());
    // thrust::copy(thrust::device, d_q.begin(), new_last, compacted_quantiles->begin() + old_size);
    }
  raft::update_device(useful_nbins->data(), h_useful_nbins.data(), n_cols, handle.get_stream());
  ML::POP_RANGE(); // sequential thrust::unique
  ML::POP_RANGE(); // Sequential
}

template <typename T>
  auto computeQuantiles(
  int n_bins, const T* data, int n_rows, int n_cols, const raft::handle_t& handle)
{
  ML::PUSH_RANGE("computeQuantiles");
  auto quantiles = std::make_shared<rmm::device_uvector<T>>(n_bins * n_cols, handle.get_stream());
  auto useful_nbins = std::make_shared<rmm::device_uvector<int>>(n_cols, handle.get_stream());
  int STRATEGY = getenv("STRATEGY")[0] - 48;
  switch(STRATEGY) {
    case 1:
    printf("calling computeQuantilesHelperV1\n");
    computeQuantilesHelperV1(data, quantiles, useful_nbins, n_bins, n_rows, n_cols, handle);
    break;
    case 2:
    printf("calling computeQuantilesHelperV2\n");
    computeQuantilesHelperV2(data, quantiles, useful_nbins, n_bins, n_rows, n_cols, handle);
    break;
    case 3:
    printf("calling computeQuantilesHelperV3\n");
    computeQuantilesHelperV3(data, quantiles, useful_nbins, n_bins, n_rows, n_cols, handle);
    break;
    case 4:
    printf("calling computeQuantilesHelperV4\n");
    computeQuantilesHelperV4(data, quantiles, useful_nbins, n_bins, n_rows, n_cols, handle);
    break;
    case 5:
    printf("calling computeQuantilesHelperV5\n");
    computeQuantilesHelperV5(data, quantiles, useful_nbins, n_bins, n_rows, n_cols, handle);
    break;
    default: printf("wrong STRATEGY!\n");
    break;
  }
  ML::POP_RANGE(); // computeQuantiles
  // debug printing
  // raft::print_device_vector("quantiles: ", quantiles->begin(), n_bins*n_cols, std::cout);
  // raft::print_device_vector("useful_nbins ", useful_nbins->begin(), n_cols, std::cout);
  return std::make_pair(quantiles, useful_nbins);
}

}  // namespace DT
}  // namespace ML
