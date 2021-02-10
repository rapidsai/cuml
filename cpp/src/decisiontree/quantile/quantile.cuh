/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <raft/cuda_utils.cuh>
#include <cub/cub.cuh>
#include "quantile.h"

namespace ML {
namespace DecisionTree {

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
                                  const int nbins, bool isVerbose=false) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nbins * ncols) {
    int binoff = (int)(nrows / nbins);
    int coloff = (int)(tid / nbins) * nrows;
    quantile[tid] = data[((tid % nbins) + 1) * binoff - 1 + coloff];
    // if(isVerbose) {
    //   float bin_width = float(nrows) / nbins;
    //   int index = ((tid % nbins) + 1) * binoff - 1 + coloff;
    //   // int binIndex = tid + 1;
    //   // printf("tid = %d, index = %d, Exact cut = %f, rounded = %d,\n", tid, 
    //   //   ((tid % nbins) + 1) * binoff - 1 + coloff,
    //   //   binIndex*binWidth - 1, int(round(binIndex*binWidth))-1);
    //   printf("[*] bin/n_bins = %d/%d, Exact cut = %f, rounded = %d,\n", tid,
    //          nbins, (tid + 1)*bin_width, index);
    // }
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
  printf("[quantile.cuh] In the preprocess_quantile()\n");
  int batch_cols =
    1;  // Processing one column at a time, for now, until an appropriate getMemInfo function is provided for the deviceAllocator interface.

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
  printf("[quantile.cuh:%d] blocks: %d, threads:%d\n", __LINE__, blocks, threads);
  set_sorting_offset<<<blocks, threads, 0, tempmem->stream>>>(
    n_sampled_rows, batch_cols, d_offsets->data());
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
  printf("[quantile.cuh:%d] n_sampled_rows: %d, n_cols:%d\n", __LINE__, n_sampled_rows, ncols);
  printf("[quantile.cuh:%d] batch_cnt: %d, last_batch_size: %d, batch_items: %d\n",
    __LINE__, batch_cnt, last_batch_size, batch_items);
  printf("[quantile.cuh:%d] nbins: %d\n", __LINE__, nbins);
  d_keys_out = new MLCommon::device_buffer<T>(tempmem->device_allocator,
                                              tempmem->stream, batch_items);

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out->data(),
    batch_items, 0, 8 * sizeof(T), tempmem->stream));

  // Allocate temporary storage
  d_temp_storage = new MLCommon::device_buffer<char>(
    tempmem->device_allocator, tempmem->stream, temp_storage_bytes);
  printf("[quantile.cuh:%d] temp_storage_bytes: %d\n", __LINE__, temp_storage_bytes);
  // Compute quantiles for cur_batch_cols columns per loop iteration.
  for (int batch = 0; batch < batch_cnt; batch++) {
    int cur_batch_cols = (batch == batch_cnt - 1)
                           ? last_batch_size
                           : batch_cols;  // properly handle the last batch

    int batch_offset = batch * n_sampled_rows * batch_cols;
    int quantile_offset = batch * nbins * batch_cols;

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      (void *)d_temp_storage->data(), temp_storage_bytes,
      &d_keys_in[batch_offset], d_keys_out->data(), n_sampled_rows, 0,
      8 * sizeof(T), tempmem->stream));

    blocks = raft::ceildiv(cur_batch_cols * nbins, threads);
    get_all_quantiles<<<blocks, threads, 0, tempmem->stream>>>(
      d_keys_out->data(), &tempmem->d_quantile->data()[quantile_offset],
      n_sampled_rows, cur_batch_cols, nbins, batch == 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  }
  raft::update_host(tempmem->h_quantile->data(), tempmem->d_quantile->data(),
                    nbins * ncols, tempmem->stream);
  d_keys_out->release(tempmem->stream);
  d_offsets->release(tempmem->stream);
  d_temp_storage->release(tempmem->stream);
  delete d_keys_out;
  delete d_offsets;
  delete d_temp_storage;

  return;
}


template <typename T>
__global__ void computeQuantilesSorted(T *quantiles, const int n_bins,
                                       const T *sorted_data, 
                                       const int length, 
                                       bool isVerbose=false) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float bin_width = float(length) / n_bins;
  int index = int(round((tid + 1)*bin_width)) - 1;
  // int index = (tid + 1)*floor(bin_width) - 1;
  if(tid < n_bins) {
    quantiles[tid] = sorted_data[index];
    // if(isVerbose) { 
    //   printf("bin/n_bins = %d/%d, Exact cut = %f, rounded = %d,\n", tid,
    //          n_bins, (tid + 1)*bin_width, index);
    // }
  }

  return;
}

template <typename T>
void computeQuantiles(
  T* quantiles, const int n_bins, const T *data,
  const int n_rows, const int n_cols,
  const std::shared_ptr<MLCommon::deviceAllocator> device_allocator,
  const std::shared_ptr<MLCommon::hostAllocator> host_allocator, 
  cudaStream_t stream) {

  printf("[quantile.cuh] In the preprocess_quantile()\n");
  printf("[quantile.cuh:%d] n_rows: %d, n_cols:%d\n", __LINE__, n_rows, n_cols);

  CUDA_CHECK(cudaGetLastError());

  // Determine temporary device storage requirements
  MLCommon::device_buffer<char> *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  printf("[quantile.cuh:%d] n_cols: %d, last_batch_size: %d, batch_items: %d\n",
    __LINE__, n_cols, 1, n_rows);

  MLCommon::device_buffer<T> *single_column_sorted;
  single_column_sorted = new MLCommon::device_buffer<T>(device_allocator, stream, n_rows);

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
    d_temp_storage, temp_storage_bytes, data, single_column_sorted->data(),
    n_rows, 0, 8 * sizeof(T), stream));

  // Allocate temporary storage for sorting
  d_temp_storage = new MLCommon::device_buffer<char>(
    device_allocator, stream, temp_storage_bytes);

  // Compute quantiles column by column
  for (int col = 0; col < n_cols; col++) {
    int col_offset = col * n_rows;
    int quantile_offset = col * n_bins;

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      (void *)d_temp_storage->data(), temp_storage_bytes,
      &data[col_offset], single_column_sorted->data(), n_rows, 0,
      8 * sizeof(T), stream));

    int blocks = raft::ceildiv(n_bins, 128);
  
    computeQuantilesSorted<<<blocks, 128, 0, stream>>>(
      &quantiles[quantile_offset], n_bins, single_column_sorted->data(), n_rows,
      col == 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  single_column_sorted->release(stream);
  d_temp_storage->release(stream);

  delete single_column_sorted;
  delete d_temp_storage;

  return;
}

}  // namespace DecisionTree
}  // namespace ML
