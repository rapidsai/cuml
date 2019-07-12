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
#include "col_condenser.cuh"
#include "cub/cub.cuh"
#include "quantile.h"

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
void preprocess_quantile(T *data, const unsigned int *rowids,
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
  int batch_cols =
    1;  // Processing one column at a time, for now, until an appropriate getMemInfo function is provided for the deviceAllocator interface.

  int threads = 128;
  MLCommon::device_buffer<int> *d_offsets;
  MLCommon::device_buffer<T> *d_keys_out;
  T *d_keys_in;
  int blocks;
  if (tempmem->temp_data != nullptr) {
    d_keys_in = tempmem->temp_data->data();
    unsigned int *colids = nullptr;
    blocks = MLCommon::ceildiv(ncols * n_sampled_rows, threads);
    allcolsampler_kernel<<<blocks, threads, 0, tempmem->stream>>>(
      data, rowids, colids, n_sampled_rows, ncols, rowoffset,
      d_keys_in);  // d_keys_in already allocated for all ncols
    CUDA_CHECK(cudaGetLastError());
  } else {
    d_keys_in = data;
  }
  
  d_offsets = new MLCommon::device_buffer<int>(
    tempmem->ml_handle.getDeviceAllocator(), tempmem->stream, batch_cols + 1);

  blocks = MLCommon::ceildiv(batch_cols + 1, threads);
  set_sorting_offset<<<blocks, threads, 0, tempmem->stream>>>(
    n_sampled_rows, batch_cols, d_offsets->data());
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

  d_keys_out = new MLCommon::device_buffer<T>(
    tempmem->ml_handle.getDeviceAllocator(), tempmem->stream, batch_items);
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out->data(),
    batch_items, batch_cols, d_offsets->data(), d_offsets->data() + 1, 0,
    8 * sizeof(T), tempmem->stream));

  // Allocate temporary storage
  d_temp_storage =
    new MLCommon::device_buffer<char>(tempmem->ml_handle.getDeviceAllocator(),
                                      tempmem->stream, temp_storage_bytes);

  // Compute quantiles for cur_batch_cols columns per loop iteration.
  for (int batch = 0; batch < batch_cnt; batch++) {
    int cur_batch_cols = (batch == batch_cnt - 1)
                           ? last_batch_size
                           : batch_cols;  // properly handle the last batch

    int batch_offset = batch * n_sampled_rows * batch_cols;
    int quantile_offset = batch * nbins * batch_cols;

    // Run sorting operation
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
      (void *)d_temp_storage->data(), temp_storage_bytes,
      &d_keys_in[batch_offset], d_keys_out->data(), n_sampled_rows * batch_cols,
      cur_batch_cols, d_offsets->data(), d_offsets->data() + 1, 0,
      8 * sizeof(T), tempmem->stream));

    blocks = MLCommon::ceildiv(cur_batch_cols * nbins, threads);
    get_all_quantiles<<<blocks, threads, 0, tempmem->stream>>>(
      d_keys_out->data(), &tempmem->d_quantile->data()[quantile_offset],
      n_sampled_rows, cur_batch_cols, nbins);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  }
  MLCommon::updateHost(tempmem->h_quantile->data(), tempmem->d_quantile->data(),
                       nbins * ncols, tempmem->stream);
  d_keys_out->release(tempmem->stream);
  d_offsets->release(tempmem->stream);
  d_temp_storage->release(tempmem->stream);
  delete d_keys_out;
  delete d_offsets;
  delete d_temp_storage;

  return;
}
