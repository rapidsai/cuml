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

#include <stdint.h>
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "cuda_utils.h"

// This file is a shameless amalgamation of independent works done by
// Lars Nyland and Andy Adinets

namespace MLCommon {
namespace Stats {

/** Default mapper which just returns the value of the data itself */
template <typename DataT, typename IdxT>
struct IdentityBinner {
  DI int operator()(DataT val, IdxT i) { return int(val); }
};

/** Types of support histogram implementations */
enum HistType {
  /** use only global atomics */
  HistTypeGmem,
  /** uses shared mem atomics to reduce global traffic */
  HistTypeSmem,
  /** shared mem atomics but with bins to be 2B int's */
  HistTypeSmemBits16,
  /** shared mem atomics but with bins to ba 1B int's */
  HistTypeSmemBits8,
  /** builds a hashmap of active bins in shared mem */
  HistTypeHash
};

template <typename DataT, typename BinnerOp, typename IdxT>
__global__ void gmemHistKernel(int* bins, const DataT* data, IdxT n,
                               BinnerOp binner) {
  auto i = threadIdx.x + IdxT(blockIdx.x) * blockDim.x;
  auto stride = IdxT(blockDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    int binId = binner(data[i], i);
    atomicAdd(bins + binId, 1);
  }
}

template <typename DataT, typename BinnerOp, typename IdxT = int, int TPB = 256>
void gmemHist(int* bins, IdxT nbins, const DataT* data, IdxT n, BinnerOp op,
              cudaStream_t stream) {
  int nblks = ceildiv<int>(n, TPB);
  CUDA_CHECK(cudaMemsetAsync(bins, 0, nbins * sizeof(int), stream));
  gmemHistKernel<DataT, BinnerOp, IdxT>
    <<<nblks, TPB, 0, stream>>>(bins, data, n, op);
  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename BinnerOp, typename IdxT>
__global__ void smemHistKernel(int* bins, const DataT* data, IdxT n,
                               BinnerOp binner) {
  extern __shared__ int sbins[];
  auto tid = threadIdx.x + IdxT(blockIdx.x) * blockDim.x;
  for (auto i = tid; i < nbins; i += blockDim.x) {
    sbins[i] = 0;
  }
  __syncthreads();
  auto stride = IdxT(blockDim.x) * blockDim.x;
  for (auto i = tid; i < n; i += stride) {
    int binId = binner(data[i], i);
    atomicAdd(sbins + binId, 1);
  }
  __syncthreads();
  for (auto i = tid; i < nbins; i += blockDim.x) {
    atomicAdd(bins + i, sbins[i]);
  }
}

template <typename DataT, typename BinnerOp, typename IdxT = int, int TPB = 256>
void smemHist(int* bins, IdxT nbins, const DataT* data, IdxT n, BinnerOp op,
              cudaStream_t stream) {
  int nblks = ceildiv<int>(n, TPB);
  CUDA_CHECK(cudaMemsetAsync(bins, 0, nbins * sizeof(int), stream));
  size_t smemSize = nbins * sizeof(int);
  smemHistKernel<DataT, BinnerOp, IdxT>
    <<<nblks, TPB, smemSize, stream>>>(bins, data, n, op);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Naive global memory atomics based histogram kernel
 * @tparam DataT input data type
 * @tparam BinnerOp takes the input data and computes its bin index
 * @tparam IdxT data type used to compute indices
 * @tparam TPB threads per block
 * @param type histogram implementation type to choose
 * @param bins the output bins (length = nbins)
 * @param nbins number of bins
 * @param data input data (length = n)
 * @param n data array length
 * @param allocator device allocator
 * @param stream cuda stream
 * @param op the operation that computes the bin index of the input data
 *
 * @note signature of BinnerOp is `int func(DataT, IdxT);`
 */
template <typename DataT, typename BinnerOp, typename IdxT = int, int TPB = 256>
void histogram(HistType type, int* bins, IdxT nbins, const DataT* data, IdxT n,
               std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream,
               BinnerOp op = IdentityBinner()) {
  switch (type) {
    case HistTypeGmem:
      gmemHist<DataT, BinnerOp, IdxT, TPB>(bins, nbins, data, n, op, stream);
      break;
    case HistTypeSmem:
      smemHist<DataT, BinnerOp, IdxT, TPB>(bins, nbins, data, n, op, stream);
      break;
    default:
      ASSERT(false, "histogram: Invalid type passed '%d'!", type);
  };
}

};  // end namespace Stats
};  // end namespace MLCommon
