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
