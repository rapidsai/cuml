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
#include "utils.h"

// This file is a shameless amalgamation of independent works done by
// Lars Nyland and Andy Adinets

///@todo: add cub's histogram as another option

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
  /** use only global atomics but with cache-line aware acceses */
  HistTypeGmemSwizzle,
  /** uses shared mem atomics to reduce global traffic */
  HistTypeSmem,
  /** shared mem atomics but with bins to be 2B int's */
  HistTypeSmemBits16,
  /** shared mem atomics but with bins to ba 1B int's */
  HistTypeSmemBits8,
  /** builds a hashmap of active bins in shared mem */
  HistTypeSmemHash,
  /** decide at runtime the best algo for the given inputs */
  HistTypeAuto
};

DI int swizzleBinId(int id) {
  // Swap bits 0:5 with 6:11 in bucket.  Moves adjacenct values farther apart.
  // Requires at least 4096 buckets.
  // Why 6?
  //   That's 64 words or 256 bytes, which is the inter-slice stride in the
  //   L2, so adjacent buckets will now be on different slices
  // Don't forget to undo the transform when you read the buckets out
  int a = id & 0x3f;
  int b = (id >> 6) & 0x3f;
  id >>= 12;
  id <<= 12;
  id |= (a << 6) | b;
  return id;
}

template <typename DataT, typename BinnerOp, typename IdxT, bool SwapBits>
__global__ void gmemHistKernel(int* bins, const DataT* data, IdxT n,
                               BinnerOp binner) {
  auto i = threadIdx.x + IdxT(blockIdx.x) * blockDim.x;
  auto stride = IdxT(blockDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    int binId = binner(data[i], i);
    if (SwapBits) {
      binId = swizzleBinId(binId);
    }
    atomicAdd(bins + binId, 1);
  }
}

///@todo: launch half the number of threads, but also consider odd number of
/// bins case while doing so
__global__ void unswapKernel(int* bins, int nbins) {
  auto src = threadIdx.x + blockDim.x * blockIdx.x;
  auto tgt = swizzleBinId(src);
  if (src >= nbins || tgt >= nbins || src >= tgt) {
    return;
  }
  auto a = bins[src];
  auto b = bins[tgt];
  bins[src] = b;
  bins[tgt] = a;
}

template <typename DataT, typename BinnerOp, typename IdxT, bool SwapBits,
          int TPB>
void gmemHist(int* bins, IdxT nbins, const DataT* data, IdxT n, BinnerOp op,
              cudaStream_t stream) {
  int nblks = ceildiv<int>(n, TPB);
  CUDA_CHECK(cudaMemsetAsync(bins, 0, nbins * sizeof(int), stream));
  gmemHistKernel<DataT, BinnerOp, IdxT, SwapBits>
    <<<nblks, TPB, 0, stream>>>(bins, data, n, op);
  CUDA_CHECK(cudaGetLastError());
  if (SwapBits) {
    unswapKernel<IdxT><<<ceildiv<int>(nbins, TPB), TPB>>>(bins, nbins);
    CUDA_CHECK(cudaGetLastError());
  }
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

template <unsigned BIN_BITS>
DI void incrementBin(int* sbins, int* bins, int nbins, int binId) {
  constexpr unsigned WORD_BITS = sizeof(int) * 8;
  constexpr unsigned WORD_BINS = WORD_BITS / BIN_BITS;
  constexpr unsigned BIN_MASK = (1 << BIN_BITS) - 1;
  auto iword = bin / WORD_BINS;
  auto ibin = bin % WORD_BINS;
  auto sh = ibin * BIN_BITS;
  auto old_word = atomicAdd(sbins + iword, 1 << sh);
  auto new_word = old_word + (1 << sh);
  if ((new_word >> sh & BIN_MASK) != 0) return;
  // overflow
  atomicAdd(&bins[bin], BIN_MASK + 1);
  for (unsigned dbin = 1; ibin + dbin < WORD_BINS && bin + dbin < nbins;
       ++dbin) {
    auto sh1 = (ibin + dbin) * BIN_BITS;
    if ((new_word >> sh1 & BIN_MASK) == 0) {
      // overflow
      atomicAdd(&bins[bin + dbin], BIN_MASK);
    } else {
      // correction
      atomicAdd(&bins[bin + dbin], -1);
      break;
    }
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, unsigned BIN_BITS>
__global__ void smemBitsHistKernel(int* bins, const DataT* data, IdxT n,
                                   IdxT nbins, BinnerOp binner) {
  extern __shared__ int sbins[];
  constexpr unsigned WORD_BITS = sizeof(int) * 8;
  constexpr unsigned WORD_BINS = WORD_BITS / BIN_BITS;
  constexpr unsigned BIN_MASK = (1 << BIN_BITS) - 1;
  auto nwords = ceildiv<int>(nbins, WORD_BINS);
  for (int j = threadIdx.x; j < nwords; j += blockDim.x) {
    sbins[j] = 0;
  }
  __syncthreads();
  IdxT tid = threadIdx.x + IdxT(blockDim.x) * blockIdx.x;
  IdxT stride = IdxT(blockDim.x) * gridDim.x;
  for (IdxT i = tid; i < n; i += stride) {
    int binId = binner(data[i], i);
    incrementBin<BIN_BITS>(sbins, bins, (int)nbins, binId);
  }
  __syncthreads();
  for (int j = threadIdx.x; j < (int)nbins; j += blockDim.x) {
    int count = sbins[j / WORD_BINS] >> (j % WORD_BINS * BIN_BITS) & BIN_MASK;
    if (count > 0) atomicAdd(bins + j, count);
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int TPB,
          unsigned BIN_BITS>
void smemBitsHist(int* bins, IdxT nbins, const DataT* data, IdxT n, BinnerOp op,
                  cudaStream_t stream) {
  int nblks = ceildiv<int>(n, TPB);
  CUDA_CHECK(cudaMemsetAsync(bins, 0, nbins * sizeof(int), stream));
  constexpr unsigned WORD_BITS = sizeof(int) * 8;
  size_t smemSize = ceildiv<size_t>(nbins, WORD_BITS / BIN_BITS) * sizeof(int);
  smemBitsHistKernel<DataT, BinnerOp, IdxT, BIN_BITS>
    <<<nblks, TPB, smemSize, stream>>>(bins, data, n, op);
  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename BinnerOp, typename IdxT, int TPB>
void histogramImpl(HistType type, int* bins, IdxT nbins, const DataT* data,
                   IdxT n, std::shared_ptr<deviceAllocator> allocator,
                   cudaStream_t stream, BinnerOp op) {
  switch (type) {
    case HistTypeGmem:
      gmemHist<DataT, BinnerOp, IdxT, false, TPB>(bins, nbins, data, n, op,
                                                  stream);
      break;
    case HistTypeGmemSwizzle:
      gmemHist<DataT, BinnerOp, IdxT, true, TPB>(bins, nbins, data, n, op,
                                                 stream);
      break;
    case HistTypeSmem:
      smemHist<DataT, BinnerOp, IdxT, TPB>(bins, nbins, data, n, op, stream);
      break;
    case HistTypeSmemBits16:
      smemBitsHist<DataT, BinnerOp, IdxT, TPB, 16>(bins, nbins, data, n, op,
                                                   stream);
      break;
    case HistTypeSmemBits8:
      smemBitsHist<DataT, BinnerOp, IdxT, TPB, 8>(bins, nbins, data, n, op,
                                                  stream);
      break;
    default:
      ASSERT(false, "histogram: Invalid type passed '%d'!", type);
  };
}

/**
 * @brief Perform histogram on the input data
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
  HistType computeType = type;
  if (type == HistTypeAuto) {
    ///@todo: implement!
  }
  histogramImpl<DataT, BinnerUp, IdxT, TPB>(computedType, bins, nbins, data, n,
                                            allocator, stream, op);
}

};  // end namespace Stats
};  // end namespace MLCommon
