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
#include "cuda_utils.h"

template <typename T>
__global__ void get_sampled_column_kernel(
  const T* __restrict__ column, T* outcolumn,
  const unsigned int* __restrict__ rowids, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int index = rowids[tid];
    outcolumn[tid] = column[index];
  }
  return;
}

template <typename T>
void get_sampled_labels(const T* labels, T* outlabels,
                        const unsigned int* rowids, const int n_sampled_rows,
                        const cudaStream_t stream) {
  int threads = 128;
  get_sampled_column_kernel<T>
    <<<MLCommon::ceildiv(n_sampled_rows, threads), threads, 0, stream>>>(
      labels, outlabels, rowids, n_sampled_rows);
  CUDA_CHECK(cudaGetLastError());
  return;
}

template <typename T>
__global__ void allcolsampler_kernel(const T* __restrict__ data,
                                     const unsigned int* __restrict__ rowids,
                                     const unsigned int* __restrict__ colids,
                                     const int nrows, const int ncols,
                                     const int rowoffset, T* sampledcols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (unsigned int i = tid; i < nrows * ncols; i += blockDim.x * gridDim.x) {
    int newcolid = (int)(i / nrows);
    int myrowstart;
    if (colids != nullptr) {
      myrowstart = colids[newcolid] * rowoffset;
    } else {
      myrowstart = newcolid * rowoffset;
    }

    int index = rowids[i % nrows] + myrowstart;
    sampledcols[i] = data[index];
  }
  return;
}
