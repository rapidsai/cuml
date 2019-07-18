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
#include "cuda_utils.h"
#include "utils.h"
#include "vectorized.h"

// This file is a shameless amalgamation of independent works done by
// Lars Nyland and Andy Adinets

///@todo: add cub's histogram as another option
///@todo: add support for hash-based smem technique

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
  HistTypeSmemHash,
  /** decide at runtime the best algo for the given inputs */
  HistTypeAuto
};

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen,
          typename CoreOp>
DI void histCoreOp(const DataT* data, IdxT n, BinnerOp binOp, CoreOp op) {
  IdxT tid = threadIdx.x + IdxT(blockDim.x) * blockIdx.x;
  tid *= VecLen;
  IdxT stride = IdxT(blockDim.x) * gridDim.x * VecLen;
  typedef TxN_t<DataT, VecLen> VecType;
  VecType a;
  for (auto i = tid; i < n; i += stride) {
    a.load(data, i);
#pragma unroll
    for (int j = 0; j < VecLen; ++j) {
      int binId = binOp(a.val.data[j], i + j);
      op(binId, i + j);
    }
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
__global__ void gmemHistKernel(int* bins, const DataT* data, IdxT n,
                               BinnerOp binner) {
  auto op = [=] __device__(int binId, IdxT idx) { atomicAdd(bins + binId, 1); };
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, n, binner, op);
}

template <typename DataT, typename BinnerOp, typename IdxT, int TPB, int VecLen>
void gmemHist(int* bins, IdxT nbins, const DataT* data, IdxT n, BinnerOp op,
              cudaStream_t stream) {
  int nblks = ceildiv<int>(VecLen ? n / VecLen : n, TPB);
  gmemHistKernel<DataT, BinnerOp, IdxT, VecLen>
    <<<nblks, TPB, 0, stream>>>(bins, data, n, op);
  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
__global__ void smemHistKernel(int* bins, const DataT* data, IdxT n, IdxT nbins,
                               BinnerOp binner) {
  extern __shared__ int sbins[];
  for (auto i = threadIdx.x; i < nbins; i += blockDim.x) {
    sbins[i] = 0;
  }
  __syncthreads();
  auto op = [=] __device__(int binId, IdxT idx) {
    atomicAdd(sbins + binId, 1);
  };
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, n, binner, op);
  __syncthreads();
  for (auto i = threadIdx.x; i < nbins; i += blockDim.x) {
    atomicAdd(bins + i, sbins[i]);
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int TPB, int VecLen>
void smemHist(int* bins, IdxT nbins, const DataT* data, IdxT n, BinnerOp op,
              cudaStream_t stream) {
  int nblks = ceildiv<int>(VecLen ? n / VecLen : n, TPB);
  size_t smemSize = nbins * sizeof(int);
  smemHistKernel<DataT, BinnerOp, IdxT, VecLen>
    <<<nblks, TPB, smemSize, stream>>>(bins, data, n, nbins, op);
  CUDA_CHECK(cudaGetLastError());
}

template <int BIN_BITS>
DI void incrementBin(int* sbins, int* bins, int nbins, int binId) {
  constexpr int WORD_BITS = sizeof(int) * 8;
  constexpr int WORD_BINS = WORD_BITS / BIN_BITS;
  constexpr int BIN_MASK = (1 << BIN_BITS) - 1;
  auto iword = binId / WORD_BINS;
  auto ibin = binId % WORD_BINS;
  auto sh = ibin * BIN_BITS;
  auto old_word = atomicAdd(sbins + iword, 1 << sh);
  auto new_word = old_word + (1 << sh);
  if ((new_word >> sh & BIN_MASK) != 0) return;
  // overflow
  atomicAdd(bins + binId, BIN_MASK + 1);
  for (int dbin = 1; ibin + dbin < WORD_BINS && binId + dbin < nbins; ++dbin) {
    auto sh1 = (ibin + dbin) * BIN_BITS;
    if ((new_word >> sh1 & BIN_MASK) == 0) {
      // overflow
      atomicAdd(bins + binId + dbin, BIN_MASK);
    } else {
      // correction
      atomicAdd(bins + binId + dbin, -1);
      break;
    }
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int BIN_BITS,
          int VecLen>
__global__ void smemBitsHistKernel(int* bins, const DataT* data, IdxT n,
                                   IdxT nbins, BinnerOp binner) {
  extern __shared__ int sbins[];
  constexpr int WORD_BITS = sizeof(int) * 8;
  constexpr int WORD_BINS = WORD_BITS / BIN_BITS;
  constexpr int BIN_MASK = (1 << BIN_BITS) - 1;
  auto nwords = ceildiv<int>(nbins, WORD_BINS);
  for (auto j = threadIdx.x; j < nwords; j += blockDim.x) {
    sbins[j] = 0;
  }
  __syncthreads();
  auto op = [=] __device__(int binId, IdxT idx) {
    incrementBin<BIN_BITS>(sbins, bins, (int)nbins, binId);
  };
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, n, binner, op);
  __syncthreads();
  for (auto j = threadIdx.x; j < (int)nbins; j += blockDim.x) {
    int count = sbins[j / WORD_BINS] >> (j % WORD_BINS * BIN_BITS) & BIN_MASK;
    if (count > 0) atomicAdd(bins + j, count);
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int TPB,
          int BIN_BITS, int VecLen>
void smemBitsHist(int* bins, IdxT nbins, const DataT* data, IdxT n, BinnerOp op,
                  cudaStream_t stream) {
  int nblks = ceildiv<int>(VecLen ? n / VecLen : n, TPB);
  constexpr int WORD_BITS = sizeof(int) * 8;
  size_t smemSize = ceildiv<size_t>(nbins, WORD_BITS / BIN_BITS) * sizeof(int);
  smemBitsHistKernel<DataT, BinnerOp, IdxT, BIN_BITS, VecLen>
    <<<nblks, TPB, smemSize, stream>>>(bins, data, n, nbins, op);
  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename BinnerOp, typename IdxT, int TPB, int VecLen>
void histogramVecLen(HistType type, int* bins, IdxT nbins, const DataT* data,
                     IdxT n, cudaStream_t stream, BinnerOp op) {
  CUDA_CHECK(cudaMemsetAsync(bins, 0, nbins * sizeof(int), stream));
  switch (type) {
    case HistTypeGmem:
      gmemHist<DataT, BinnerOp, IdxT, TPB, VecLen>(bins, nbins, data, n, op,
                                                   stream);
      break;
    case HistTypeSmem:
      smemHist<DataT, BinnerOp, IdxT, TPB, VecLen>(bins, nbins, data, n, op,
                                                   stream);
      break;
    case HistTypeSmemBits16:
      smemBitsHist<DataT, BinnerOp, IdxT, TPB, 16, VecLen>(bins, nbins, data, n,
                                                           op, stream);
      break;
    case HistTypeSmemBits8:
      smemBitsHist<DataT, BinnerOp, IdxT, TPB, 8, VecLen>(bins, nbins, data, n,
                                                          op, stream);
      break;
    case HistTypeSmemHash:
      ///@todo!!
      break;
    default:
      ASSERT(false, "histogram: Invalid type passed '%d'!", type);
  };
}

template <typename DataT, typename BinnerOp, typename IdxT, int TPB>
void histogramImpl(HistType type, int* bins, IdxT nbins, const DataT* data,
                   IdxT n, cudaStream_t stream, BinnerOp op) {
  size_t bytes = n * sizeof(DataT);
  if (n <= 0) return;
  if (16 / sizeof(DataT) && bytes % 16 == 0) {
    histogramVecLen<DataT, BinnerOp, IdxT, TPB, 16 / sizeof(DataT)>(
      type, bins, nbins, data, n, stream, op);
  } else if (8 / sizeof(DataT) && bytes % 8 == 0) {
    histogramVecLen<DataT, BinnerOp, IdxT, TPB, 8 / sizeof(DataT)>(
      type, bins, nbins, data, n, stream, op);
  } else if (4 / sizeof(DataT) && bytes % 4 == 0) {
    histogramVecLen<DataT, BinnerOp, IdxT, TPB, 4 / sizeof(DataT)>(
      type, bins, nbins, data, n, stream, op);
  } else if (2 / sizeof(DataT) && bytes % 2 == 0) {
    histogramVecLen<DataT, BinnerOp, IdxT, TPB, 2 / sizeof(DataT)>(
      type, bins, nbins, data, n, stream, op);
  } else {
    histogramVecLen<DataT, BinnerOp, IdxT, TPB, 1>(type, bins, nbins, data, n,
                                                   stream, op);
  }
}

template <typename IdxT>
HistType selectBestHistAlgo(IdxT nbins) {
  size_t smem = maxSharedMem();
  size_t requiredSize = nbins * sizeof(int);
  if (requiredSize <= smem) {
    return HistTypeSmem;
  }
  requiredSize = ceildiv<size_t>(nbins, 2) * sizeof(int);
  if (requiredSize <= smem) {
    return HistTypeSmemBits16;
  }
  requiredSize = ceildiv<size_t>(nbins, 4) * sizeof(int);
  if (requiredSize <= smem) {
    return HistTypeSmemBits8;
  }
  return HistTypeGmem;
}

/**
 * @brief Perform histogram on the input data. It chooses the right load size
 * based on the input data vector length. It also supports large-bin cases
 * using a specialized smem-based hashing technique.
 * @tparam DataT input data type
 * @tparam IdxT data type used to compute indices
 * @tparam TPB threads per block
 * @tparam BinnerOp takes the input data and computes its bin index
 * @param type histogram implementation type to choose
 * @param bins the output bins (length = nbins)
 * @param nbins number of bins
 * @param data input data (length = n)
 * @param n data array length
 * @param stream cuda stream
 * @param op the operation that computes the bin index of the input data
 *
 * @note signature of BinnerOp is `int func(DataT, IdxT);`
 */
template <typename DataT, typename IdxT = int, int TPB = 256,
          typename BinnerOp = IdentityBinner<DataT, IdxT>>
void histogram(HistType type, int* bins, IdxT nbins, const DataT* data, IdxT n,
               cudaStream_t stream,
               BinnerOp op = IdentityBinner<DataT, IdxT>()) {
  HistType computedType = type;
  if (type == HistTypeAuto) {
    computedType = selectBestHistAlgo(nbins);
  }
  histogramImpl<DataT, BinnerOp, IdxT, TPB>(computedType, bins, nbins, data, n,
                                            stream, op);
}

};  // end namespace Stats
};  // end namespace MLCommon
