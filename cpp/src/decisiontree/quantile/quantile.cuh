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
#include <cub/cub.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>
#include "quantile.h"

#include <common/nvtx.hpp>

namespace ML {
namespace DecisionTree {

using device_allocator = raft::mr::device::allocator;
template <typename T>
using device_buffer = raft::mr::device::buffer<T>;

template <typename T>
__global__ void allcolsampler_kernel(const T *__restrict__ data,
                                     const unsigned int *__restrict__ rowids,
                                     const unsigned int *__restrict__ colids,
                                     const int nrows, const int ncols,
                                     const int rowoffset, T *sampledcols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (unsigned int i = tid; i < nrows * ncols; i += blockDim.x * gridDim.x) {
    int newcolid = (int)(i / nrows);
    int myrowstart;
    if (colids != nullptr) {
      myrowstart = colids[newcolid] * rowoffset;
    } else {
      myrowstart = newcolid * rowoffset;
    }

    int index;
    if (rowids != nullptr) {
      index = rowids[i % nrows] + myrowstart;
    } else {
      index = i % nrows + myrowstart;
    }
    sampledcols[i] = data[index];
  }
  return;
}

__global__ void set_sorting_offset(const int nrows, const int ncols,
                                   int *offsets) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid <= ncols) offsets[tid] = tid * nrows;

  return;
}

template <typename T>
__global__ void get_all_quantiles(const T *__restrict__ data, T *quantile,
                                  const int nrows, const int ncols,
                                  const int nbins) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nbins * ncols) {
    int binoff = (int)(nrows / nbins);
    int coloff = (int)(tid / nbins) * nrows;
    quantile[tid] = data[((tid % nbins) + 1) * binoff - 1 + coloff];
  }
  return;
}

template <typename T, typename L>
void preprocess_quantile(const T *data, const unsigned int *rowids,
                         const int n_sampled_rows, const int ncols,
                         const int rowoffset, const int nbins,
                         std::shared_ptr<TemporaryMemory<T, L>> tempmem) {
  /*
	// Dynamically determine batch_cols (number of columns processed per loop iteration) from the available device memory.
	size_t free_mem, total_mem;
	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
	int max_ncols = free_mem / (2 * n_sampled_rows * sizeof(T));
	int batch_cols = (max_ncols > ncols) ? ncols : max_ncols;
	ASSERT(max_ncols != 0, "Cannot preprocess quantiles due to insufficient device memory.");
  */

  ML::PUSH_RANGE("preprocessing quantile @quantile.cuh");
  int batch_cols =
    1;  // Processing one column at a time, for now, until an appropriate getMemInfo function is provided for the raft::mr::device::allocator interface.

  int threads = 128;
  MLCommon::device_buffer<int> *d_offsets;
  MLCommon::device_buffer<T> *d_keys_out;
  const T *d_keys_in;
  int blocks;
  if (tempmem->temp_data != nullptr) {
    T *d_keys_out = tempmem->temp_data->data();
    unsigned int *colids = nullptr;
    blocks = raft::ceildiv(ncols * n_sampled_rows, threads);
    allcolsampler_kernel<<<blocks, threads, 0, tempmem->stream>>>(
      data, rowids, colids, n_sampled_rows, ncols, rowoffset,
      d_keys_out);  // d_keys_in already allocated for all ncols
    CUDA_CHECK(cudaGetLastError());
    d_keys_in = d_keys_out;
  } else {
    d_keys_in = data;
  }

  d_offsets = new MLCommon::device_buffer<int>(tempmem->device_allocator,
                                               tempmem->stream, batch_cols + 1);

  blocks = raft::ceildiv(batch_cols + 1, threads);
  ML::PUSH_RANGE("set_sorting_offset kernel @quantile.cuh");
  set_sorting_offset<<<blocks, threads, 0, tempmem->stream>>>(
    n_sampled_rows, batch_cols, d_offsets->data());
  ML::POP_RANGE();
  CUDA_CHECK(cudaGetLastError());

  // Determine temporary device storage requirements
  MLCommon::device_buffer<char> *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  int batch_cnt =
    raft::ceildiv(ncols, batch_cols);  // number of loop iterations
  int last_batch_size =
    ncols - batch_cols * (batch_cnt - 1);  // number of columns in last batch
  int batch_items =
    n_sampled_rows * batch_cols;  // used to determine d_temp_storage size

  d_keys_out = new MLCommon::device_buffer<T>(tempmem->device_allocator,
                                              tempmem->stream, batch_items);
  ML::PUSH_RANGE(
    "DecisionTree::cub::DeviceRadixSort::SortKeys over batch_items "
    "@quantile.cuh");
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out->data(),
    batch_items, 0, 8 * sizeof(T), tempmem->stream));
  ML::POP_RANGE();
  // Allocate temporary storage
  d_temp_storage = new MLCommon::device_buffer<char>(
    tempmem->device_allocator, tempmem->stream, temp_storage_bytes);

  ML::PUSH_RANGE("iterative quantile computation for each batch");
  // Compute quantiles for cur_batch_cols columns per loop iteration.
  for (int batch = 0; batch < batch_cnt; batch++) {
    int cur_batch_cols = (batch == batch_cnt - 1)
                           ? last_batch_size
                           : batch_cols;  // properly handle the last batch

    int batch_offset = batch * n_sampled_rows * batch_cols;
    int quantile_offset = batch * nbins * batch_cols;
    ML::PUSH_RANGE("DeviceRadixSort::SortKeys");
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      (void *)d_temp_storage->data(), temp_storage_bytes,
      &d_keys_in[batch_offset], d_keys_out->data(), n_sampled_rows, 0,
      8 * sizeof(T), tempmem->stream));
    ML::POP_RANGE();

    blocks = raft::ceildiv(cur_batch_cols * nbins, threads);
    ML::PUSH_RANGE("get_all_quantiles kernel @quantile.cuh");
    get_all_quantiles<<<blocks, threads, 0, tempmem->stream>>>(
      d_keys_out->data(), &tempmem->d_quantile->data()[quantile_offset],
      n_sampled_rows, cur_batch_cols, nbins);
    ML::POP_RANGE();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  }
  ML::POP_RANGE();
  raft::update_host(tempmem->h_quantile->data(), tempmem->d_quantile->data(),
                    nbins * ncols, tempmem->stream);
  d_keys_out->release(tempmem->stream);
  d_offsets->release(tempmem->stream);
  d_temp_storage->release(tempmem->stream);
  delete d_keys_out;
  delete d_offsets;
  delete d_temp_storage;
  ML::POP_RANGE();

  return;
}

template <typename T>
__global__ void computeQuantilesSorted(T *quantiles, const int n_bins,
                                       const T *sorted_data, const int length) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double bin_width = static_cast<double>(length) / n_bins;
  int index = int(round((tid + 1) * bin_width)) - 1;
  // Old way of computing quantiles. Kept here for comparison.
  // To be deleted eventually
  // int index = (tid + 1) * floor(bin_width) - 1;
  if (tid < n_bins) {
    quantiles[tid] = sorted_data[index];
  }

  return;
}

template <typename T>
void computeQuantiles(
  T *quantiles, int n_bins, const T *data, int n_rows, int n_cols,
  const std::shared_ptr<raft::mr::device::allocator> device_allocator,
  cudaStream_t stream) {
  // Determine temporary device storage requirements
  std::unique_ptr<device_buffer<char>> d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  std::unique_ptr<device_buffer<T>> single_column_sorted = nullptr;
  single_column_sorted =
    std::make_unique<device_buffer<T>>(device_allocator, stream, n_rows);

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, data,
                                            single_column_sorted->data(),
                                            n_rows, 0, 8 * sizeof(T), stream));

  // Allocate temporary storage for sorting
  d_temp_storage = std::make_unique<device_buffer<char>>(
    device_allocator, stream, temp_storage_bytes);

  // Compute quantiles column by column
  for (int col = 0; col < n_cols; col++) {
    int col_offset = col * n_rows;
    int quantile_offset = col * n_bins;

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      (void *)d_temp_storage->data(), temp_storage_bytes, &data[col_offset],
      single_column_sorted->data(), n_rows, 0, 8 * sizeof(T), stream));

    int blocks = raft::ceildiv(n_bins, 128);

    computeQuantilesSorted<<<blocks, 128, 0, stream>>>(
      &quantiles[quantile_offset], n_bins, single_column_sorted->data(),
      n_rows);

    CUDA_CHECK(cudaGetLastError());
  }

  return;
}

}  // namespace DecisionTree
}  // namespace ML
