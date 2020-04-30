/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cuml/cuml.hpp>
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
                                  const int nbins) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nbins * ncols) {
    int binoff = (int)(nrows / nbins);
    int coloff = (int)(tid / nbins) * nrows;
    quantile[tid] = data[((tid % nbins) + 1) * binoff - 1 + coloff];
  }
  return;
}

template <typename T>
void preprocess_quantile(
  const T *data, const unsigned int *rowids, int n_sampled_rows, int ncols,
  int rowoffset, int nbins, T *h_quantile, T *d_quantile, T *temp_data,
  std::shared_ptr<MLCommon::deviceAllocator> device_allocator,
  cudaStream_t stream) {
  /*
	// Dynamically determine batch_cols (number of columns processed per loop iteration) from the available device memory.
	size_t free_mem, total_mem;
	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
	int max_ncols = free_mem / (2 * n_sampled_rows * sizeof(T));
	int batch_cols = (max_ncols > ncols) ? ncols : max_ncols;
	ASSERT(max_ncols != 0, "Cannot preprocess quantiles due to insufficient device memory.");
	*/
  int batch_cols =
    1;  // Processing one column at a time, for now, until an appropriate getMemInfo function is provided for the deviceAllocator interface.

  int threads = 128;
  MLCommon::device_buffer<int> *d_offsets;
  const T *d_keys_in;
  int blocks;
  if (temp_data != nullptr) {
    T *d_keys_out = temp_data;
    unsigned int *colids = nullptr;
    blocks = MLCommon::ceildiv(ncols * n_sampled_rows, threads);
    allcolsampler_kernel<<<blocks, threads, 0, stream>>>(
      data, rowids, colids, n_sampled_rows, ncols, rowoffset,
      d_keys_out);  // d_keys_in already allocated for all ncols
    CUDA_CHECK(cudaGetLastError());
    d_keys_in = d_keys_out;
  } else {
    d_keys_in = data;
  }

  d_offsets =
    new MLCommon::device_buffer<int>(device_allocator, stream, batch_cols + 1);

  blocks = MLCommon::ceildiv(batch_cols + 1, threads);
  set_sorting_offset<<<blocks, threads, 0, stream>>>(n_sampled_rows, batch_cols,
                                                     d_offsets->data());
  CUDA_CHECK(cudaGetLastError());

  // Determine temporary device storage requirements
  MLCommon::device_buffer<char> *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  int batch_cnt =
    MLCommon::ceildiv(ncols, batch_cols);  // number of loop iterations
  int last_batch_size =
    ncols - batch_cols * (batch_cnt - 1);  // number of columns in last batch
  int batch_items =
    n_sampled_rows * batch_cols;  // used to determine d_temp_storage size

  auto *d_keys_out =
    (T *)device_allocator->allocate(sizeof(T) * batch_items, stream);
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, batch_items,
    batch_cols, d_offsets->data(), d_offsets->data() + 1, 0, 8 * sizeof(T),
    stream));

  // Allocate temporary storage
  d_temp_storage = new MLCommon::device_buffer<char>(device_allocator, stream,
                                                     temp_storage_bytes);

  // Compute quantiles for cur_batch_cols columns per loop iteration.
  for (int batch = 0; batch < batch_cnt; batch++) {
    int cur_batch_cols = (batch == batch_cnt - 1)
                           ? last_batch_size
                           : batch_cols;  // properly handle the last batch

    int batch_offset = batch * n_sampled_rows * batch_cols;
    int quantile_offset = batch * nbins * batch_cols;

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      (void *)d_temp_storage->data(), temp_storage_bytes,
      &d_keys_in[batch_offset], d_keys_out, n_sampled_rows * batch_cols,
      cur_batch_cols, d_offsets->data(), d_offsets->data() + 1, 0,
      8 * sizeof(T), stream));

    blocks = MLCommon::ceildiv(cur_batch_cols * nbins, threads);
    get_all_quantiles<<<blocks, threads, 0, stream>>>(
      d_keys_out, &d_quantile[quantile_offset], n_sampled_rows, cur_batch_cols,
      nbins);

    CUDA_CHECK(cudaGetLastError());
  }
  if (h_quantile != nullptr) {
    MLCommon::updateHost(h_quantile, d_quantile, nbins * ncols, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  device_allocator->deallocate(d_keys_out, sizeof(T) * batch_items, stream);
  d_offsets->release(stream);
  d_temp_storage->release(stream);
  delete d_offsets;
  delete d_temp_storage;

  return;
}

}  // namespace DecisionTree
}  // namespace ML
