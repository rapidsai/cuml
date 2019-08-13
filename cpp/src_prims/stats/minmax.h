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

#include <limits>
#include "cuda_utils.h"

namespace MLCommon {
namespace Stats {

template <typename T>
struct encode_traits {};

template <>
struct encode_traits<float> {
  using E = int;
};

template <>
struct encode_traits<double> {
  using E = long long;
};

__host__ DI int encode(float val) {
  int i = *(int*)&val;
  return i >= 0 ? i : (1 << 31) | ~i;
}

__host__ DI long long encode(double val) {
  long long i = *(long long*)&val;
  return i >= 0 ? i : (1ULL << 63) | ~i;
}

__host__ DI float decode(int val) {
  if (val < 0) val = (1 << 31) | ~val;
  return *(float*)&val;
}

__host__ DI double decode(long long val) {
  if (val < 0) val = (1ULL << 63) | ~val;
  return *(double*)&val;
}

template <typename T, typename E>
DI T atomicMaxBits(T* address, T val) {
  E old = atomicMax((E*)address, encode(val));
  return decode(old);
}

template <typename T, typename E>
DI T atomicMinBits(T* address, T val) {
  E old = atomicMin((E*)address, encode(val));
  return decode(old);
}

template <typename T, typename E>
__global__ void decodeKernel(T* globalmin, T* globalmax, int ncols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < ncols) {
    globalmin[tid] = decode(*(E*)&globalmin[tid]);
    globalmax[tid] = decode(*(E*)&globalmax[tid]);
  }
}

///@todo: implement a proper "fill" kernel
template <typename T, typename E>
__global__ void minmaxInitKernel(int ncols, T* globalmin, T* globalmax,
                                 T init_val) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= ncols) return;
  *(E*)&globalmin[tid] = encode(init_val);
  *(E*)&globalmax[tid] = encode(-init_val);
}

template <typename T, typename E>
__global__ void minmaxKernel(const T* data, const unsigned int* rowids,
                             const unsigned int* colids, int nrows, int ncols,
                             int row_stride, T* g_min, T* g_max, T* sampledcols,
                             T init_min_val, int batch_ncols, int num_batches) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  extern __shared__ char shmem[];
  T* s_min = (T*)shmem;
  T* s_max = (T*)(shmem + sizeof(T) * batch_ncols);

  int last_batch_ncols = ncols % batch_ncols;
  if (last_batch_ncols == 0) {
    last_batch_ncols = batch_ncols;
  }
  int orig_batch_ncols = batch_ncols;

  for (int batch_id = 0; batch_id < num_batches; batch_id++) {
    if (batch_id == num_batches - 1) {
      batch_ncols = last_batch_ncols;
    }

    for (int i = threadIdx.x; i < batch_ncols; i += blockDim.x) {
      *(E*)&s_min[i] = encode(init_min_val);
      *(E*)&s_max[i] = encode(-init_min_val);
    }
    __syncthreads();

    for (int i = tid; i < nrows * batch_ncols; i += blockDim.x * gridDim.x) {
      int col = (batch_id * orig_batch_ncols) + (i / nrows);
      int row = i % nrows;
      if (colids != nullptr) {
        col = colids[col];
      }
      if (rowids != nullptr) {
        row = rowids[row];
      }
      int index = row + col * row_stride;
      T coldata = data[index];
      if (!isnan(coldata)) {
        //Min max values are saved in shared memory and global memory as per the shuffled colids.
        atomicMinBits<T, E>(&s_min[(int)(i / nrows)], coldata);
        atomicMaxBits<T, E>(&s_max[(int)(i / nrows)], coldata);
      }
      if (sampledcols != nullptr) {
        sampledcols[batch_id * orig_batch_ncols + i] = coldata;
      }
    }
    __syncthreads();

    // finally, perform global mem atomics
    for (int j = threadIdx.x; j < batch_ncols; j += blockDim.x) {
      atomicMinBits<T, E>(&g_min[batch_id * orig_batch_ncols + j],
                          decode(*(E*)&s_min[j]));
      atomicMaxBits<T, E>(&g_max[batch_id * orig_batch_ncols + j],
                          decode(*(E*)&s_max[j]));
    }
    __syncthreads();
  }
}

/**
 * @brief Computes min/max across every column of the input matrix, as well as
 * optionally allow to subsample based on the given row/col ID mapping vectors
 *
 * @tparam T the data type
 * @tparam TPB number of threads per block
 * @param data input data
 * @param rowids actual row ID mappings. It is of length nrows. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param colids actual col ID mappings. It is of length ncols. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param nrows number of rows of data to be worked upon. The actual rows of the
 * input "data" can be bigger than this!
 * @param ncols number of cols of data to be worked upon. The actual cols of the
 * input "data" can be bigger than this!
 * @param row_stride stride (in number of elements) between 2 adjacent columns
 * @param globalmin final col-wise global minimum (size = ncols)
 * @param globalmax final col-wise global maximum (size = ncols)
 * @param sampledcols output sampled data. Pass nullptr if you don't need this
 * @param init_val initial minimum value to be
 * @param prop cuda device properties. This is being explicitly requested since
 * querying it inside prims is not good for perf.
 * @param stream: cuda stream
 * @note This method makes the following assumptions:
 * 1. input and output matrices are assumed to be col-major
 * 2. ncols is small enough to fit the whole of min/max values across all cols
 *    in shared memory
 */
template <typename T, int TPB = 512>
void minmax(const T* data, const unsigned* rowids, const unsigned* colids,
            int nrows, int ncols, int row_stride, T* globalmin, T* globalmax,
            T* sampledcols, const cudaDeviceProp& prop, cudaStream_t stream) {
  using E = typename encode_traits<T>::E;
  int nblks = ceildiv(ncols, TPB);
  T init_val = std::numeric_limits<T>::max();
  minmaxInitKernel<T, E>
    <<<nblks, TPB, 0, stream>>>(ncols, globalmin, globalmax, init_val);
  CUDA_CHECK(cudaPeekAtLastError());
  nblks = ceildiv(nrows * ncols, TPB);
  nblks = min(nblks, 65536);
  size_t smemSize = sizeof(T) * 2 * ncols;

  // Compute the batch_ncols, in [1, ncols] range, that meet the available
  // shared memory constraints.
  int batch_ncols = min(ncols, (int)(prop.sharedMemPerBlock / (sizeof(T) * 2)));
  int num_batches = ceildiv(ncols, batch_ncols);
  smemSize = sizeof(T) * 2 * batch_ncols;

  minmaxKernel<T, E><<<nblks, TPB, smemSize, stream>>>(
    data, rowids, colids, nrows, ncols, row_stride, globalmin, globalmax,
    sampledcols, init_val, batch_ncols, num_batches);
  CUDA_CHECK(cudaPeekAtLastError());
  decodeKernel<T, E><<<nblks, TPB, 0, stream>>>(globalmin, globalmax, ncols);
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // end namespace Stats
};  // end namespace MLCommon
