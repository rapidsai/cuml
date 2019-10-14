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

namespace MLCommon {
namespace Stats {

/** Default mapper which just returns the value of the data itself */
template <typename DataT, typename IdxT>
struct IdentityBinner {
  DI int operator()(DataT val, IdxT row, IdxT col) { return int(val); }
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
  /** shared mem atomics but with bins to be 4b int's */
  HistTypeSmemBits4,
  /** shared mem atomics but with bins to be 2b int's */
  HistTypeSmemBits2,
  /** shared mem atomics but with bins to be 1b int's */
  HistTypeSmemBits1,
  /** builds a hashmap of active bins in shared mem */
  HistTypeSmemHash,
  /** decide at runtime the best algo for the given inputs */
  HistTypeAuto
};

static const int ThreadsPerBlock = 256;

template <typename IdxT, int VecLen>
dim3 computeGridDim(IdxT nrows, IdxT ncols, const void* kernel) {
  int occupancy;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel,
                                                           ThreadsPerBlock, 0));
  const auto maxBlks = occupancy * getMultiProcessorCount();
  int nblksx = ceildiv<int>(VecLen ? nrows / VecLen : nrows, ThreadsPerBlock);
  // for cases when there aren't a lot of blocks for computing one histogram
  nblksx = std::min(nblksx, maxBlks);
  return dim3(nblksx, ncols);
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen,
          typename CoreOp>
DI void histCoreOp(const DataT* data, IdxT nrows, IdxT nbins, BinnerOp binner,
                   CoreOp op, IdxT col) {
  IdxT offset = col * nrows;
  auto bdim = IdxT(blockDim.x);
  IdxT tid = threadIdx.x + bdim * blockIdx.x;
  tid *= VecLen;
  IdxT stride = bdim * gridDim.x * VecLen;
  int nCeil = alignTo<int>(nrows, stride);
  typedef TxN_t<DataT, VecLen> VecType;
  VecType a;
  a.fill(DataT(0));
  for (auto i = tid; i < nCeil; i += stride) {
    if (i < nrows) {
      a.load(data, offset + i);
    }
#pragma unroll
    for (int j = 0; j < VecLen; ++j) {
      int binId = binner(a.val.data[j], i + j, col);
      op(binId, i + j, col);
    }
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
__global__ void gmemHistKernel(int* bins, const DataT* data, IdxT nrows,
                               IdxT nbins, BinnerOp binner) {
  auto op = [=] __device__(int binId, IdxT row, IdxT col) {
    if (row >= nrows) return;
    auto binOffset = col * nbins;
#if __CUDA_ARCH__ < 700
    atomicAdd(bins + binOffset + binId, 1);
#else
    auto amask = __activemask();
    auto mask = __match_any_sync(amask, binId);
    auto leader = __ffs(mask) - 1;
    if (laneId() == leader) {
      atomicAdd(bins + binOffset + binId, __popc(mask));
    }
#endif  // __CUDA_ARCH__
  };
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, nrows, nbins, binner, op,
                                            blockIdx.y);
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
void gmemHist(int* bins, IdxT nbins, const DataT* data, IdxT nrows, IdxT ncols,
              BinnerOp binner, cudaStream_t stream) {
  auto blks = computeGridDim<IdxT, VecLen>(
    nrows, ncols, (const void*)gmemHistKernel<DataT, BinnerOp, IdxT, VecLen>);
  gmemHistKernel<DataT, BinnerOp, IdxT, VecLen>
    <<<blks, ThreadsPerBlock, 0, stream>>>(bins, data, nrows, nbins, binner);
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
__global__ void smemHistKernel(int* bins, const DataT* data, IdxT nrows,
                               IdxT nbins, BinnerOp binner) {
  extern __shared__ unsigned sbins[];
  for (auto i = threadIdx.x; i < nbins; i += blockDim.x) {
    sbins[i] = 0;
  }
  __syncthreads();
  auto op = [=] __device__(int binId, IdxT row, IdxT col) {
    if (row >= nrows) return;
#if __CUDA_ARCH__ < 700
    atomicAdd(sbins + binId, 1);
#else
    auto amask = __activemask();
    auto mask = __match_any_sync(amask, binId);
    auto leader = __ffs(mask) - 1;
    if (laneId() == leader) {
      atomicAdd(sbins + binId, __popc(mask));
    }
#endif  // __CUDA_ARCH__
  };
  IdxT col = blockIdx.y;
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, nrows, nbins, binner, op,
                                            col);
  __syncthreads();
  auto binOffset = col * nbins;
  for (auto i = threadIdx.x; i < nbins; i += blockDim.x) {
    atomicAdd(bins + binOffset + i, sbins[i]);
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
void smemHist(int* bins, IdxT nbins, const DataT* data, IdxT nrows, IdxT ncols,
              BinnerOp binner, cudaStream_t stream) {
  auto blks = computeGridDim<IdxT, VecLen>(
    nrows, ncols, (const void*)smemHistKernel<DataT, BinnerOp, IdxT, VecLen>);
  size_t smemSize = nbins * sizeof(unsigned);
  smemHistKernel<DataT, BinnerOp, IdxT, VecLen>
    <<<blks, ThreadsPerBlock, smemSize, stream>>>(bins, data, nrows, nbins,
                                                  binner);
}

template <unsigned BIN_BITS>
DI void incrementBin(unsigned* sbins, int* bins, int nbins, int binId,
                     int col) {
  constexpr unsigned WORD_BITS = sizeof(unsigned) * 8;
  constexpr unsigned WORD_BINS = WORD_BITS / BIN_BITS;
  constexpr unsigned BIN_MASK = (1 << BIN_BITS) - 1;
  auto iword = binId / WORD_BINS;
  auto ibin = binId % WORD_BINS;
  auto sh = ibin * BIN_BITS;
  auto old_word = atomicAdd(sbins + iword, unsigned(1 << sh));
  auto new_word = old_word + unsigned(1 << sh);
  if ((new_word >> sh & BIN_MASK) != 0) return;
  int binOffset = col * nbins;
  // overflow
  atomicAdd(bins + binOffset + binId, BIN_MASK + 1);
  for (int dbin = 1; ibin + dbin < WORD_BINS && binId + dbin < nbins; ++dbin) {
    auto sh1 = (ibin + dbin) * BIN_BITS;
    if ((new_word >> sh1 & BIN_MASK) == 0) {
      // overflow
      atomicAdd(bins + binOffset + binId + dbin, BIN_MASK);
    } else {
      // correction
      atomicAdd(bins + binOffset + binId + dbin, -1);
      break;
    }
  }
}

template <>
DI void incrementBin<1>(unsigned* sbins, int* bins, int nbins, int binId,
                        int col) {
  constexpr unsigned WORD_BITS = 32;
  auto iword = binId / WORD_BITS;
  auto sh = binId % WORD_BITS;
  auto old_word = atomicXor(sbins + iword, unsigned(1 << sh));
  int binOffset = col * nbins;
  if ((old_word >> sh & 1) != 0) atomicAdd(bins + binOffset + binId, 2);
}

template <typename DataT, typename BinnerOp, typename IdxT, int BIN_BITS,
          int VecLen>
__global__ void smemBitsHistKernel(int* bins, const DataT* data, IdxT nrows,
                                   IdxT nbins, BinnerOp binner) {
  extern __shared__ unsigned sbins[];
  constexpr unsigned WORD_BITS = sizeof(unsigned) * 8;
  constexpr unsigned WORD_BINS = WORD_BITS / BIN_BITS;
  constexpr unsigned BIN_MASK = (1 << BIN_BITS) - 1;
  auto nwords = ceildiv<int>(nbins, WORD_BINS);
  for (auto j = threadIdx.x; j < nwords; j += blockDim.x) {
    sbins[j] = 0;
  }
  __syncthreads();
  IdxT col = blockIdx.y;
  auto op = [=] __device__(int binId, IdxT row, IdxT col) {
    if (row >= nrows) return;
    incrementBin<BIN_BITS>(sbins, bins, (int)nbins, binId, col);
  };
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, nrows, nbins, binner, op,
                                            col);
  __syncthreads();
  IdxT binOffset = col * nbins;
  for (auto j = threadIdx.x; j < (int)nbins; j += blockDim.x) {
    int count = sbins[j / WORD_BINS] >> (j % WORD_BINS * BIN_BITS) & BIN_MASK;
    if (count > 0) atomicAdd(bins + binOffset + j, count);
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int BIN_BITS,
          int VecLen>
void smemBitsHist(int* bins, IdxT nbins, const DataT* data, IdxT nrows,
                  IdxT ncols, BinnerOp binner, cudaStream_t stream) {
  auto blks = computeGridDim<IdxT, VecLen>(
    nrows, ncols,
    (const void*)smemBitsHistKernel<DataT, BinnerOp, IdxT, BIN_BITS, VecLen>);
  constexpr int WORD_BITS = sizeof(int) * 8;
  size_t smemSize = ceildiv<size_t>(nbins, WORD_BITS / BIN_BITS) * sizeof(int);
  smemBitsHistKernel<DataT, BinnerOp, IdxT, BIN_BITS, VecLen>
    <<<blks, ThreadsPerBlock, smemSize, stream>>>(bins, data, nrows, nbins,
                                                  binner);
}

#define INVALID_KEY -1
DI void clearHashTable(int2* ht, int hashSize) {
  for (auto i = threadIdx.x; i < hashSize; i += blockDim.x) {
    ht[i] = {INVALID_KEY, 0};
  }
}

DI int findEntry(int2* ht, int hashSize, int binId) {
  int idx = binId % hashSize;
  int t;
  int count = 0;
  while ((t = atomicCAS(&(ht[idx].x), INVALID_KEY, binId)) != INVALID_KEY &&
         t != binId) {
    ++count;
    if (count >= hashSize) {
      idx = INVALID_KEY;
      break;
    }
    ++idx;
    if (idx >= hashSize) {
      idx = 0;
    }
  }
  return idx;
}

DI void flushHashTable(int2* ht, int hashSize, int* bins, int nbins, int col) {
  __syncthreads();
  int binOffset = col * nbins;
  for (auto i = threadIdx.x; i < hashSize; i += blockDim.x) {
    if (ht[i].x != INVALID_KEY && ht[i].y > 0) {
      atomicAdd(bins + binOffset + ht[i].x, ht[i].y);
    }
    ht[i] = {INVALID_KEY, 0};
  }
  __syncthreads();
}
#undef INVALID_KEY

///@todo: honor VecLen template param
template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
__global__ void smemHashHistKernel(int* bins, const DataT* data, IdxT nrows,
                                   IdxT nbins, BinnerOp binner, int hashSize) {
  extern __shared__ int2 ht[];
  int* needFlush = (int*)&(ht[hashSize]);
  if (threadIdx.x == 0) {
    needFlush[0] = 0;
  }
  clearHashTable(ht, hashSize);
  __syncthreads();
  auto op = [=] __device__(int binId, IdxT row, IdxT col) {
    bool iNeedFlush = false;
    if (row < nrows) {
      int hidx = findEntry(ht, hashSize, binId);
      if (hidx >= 0) {
        atomicAdd(&(ht[hidx].y), 1);
      } else {
        needFlush[0] = 1;
        iNeedFlush = true;
      }
    }
    __syncthreads();
    if (needFlush[0]) {
      flushHashTable(ht, hashSize, bins, nbins, col);
      if (threadIdx.x == 0) {
        needFlush[0] = 0;
      }
      __syncthreads();
    }
    if (iNeedFlush) {
      int hidx = findEntry(ht, hashSize, binId);
      atomicAdd(&(ht[hidx].y), 1);
    }
  };
  IdxT col = blockIdx.y;
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, nrows, nbins, binner, op,
                                            col);
  __syncthreads();
  flushHashTable(ht, hashSize, bins, nbins, col);
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
void smemHashHist(int* bins, IdxT nbins, const DataT* data, IdxT nrows,
                  IdxT ncols, BinnerOp binner, cudaStream_t stream) {
  auto blks = computeGridDim<IdxT, 1>(
    nrows, ncols, (const void*)smemHashHistKernel<DataT, BinnerOp, IdxT, 1>);
  // NOTE: assumes 48kB smem!
  int hashSize = 6047;
  size_t smemSize = hashSize * sizeof(int2) + sizeof(int);
  smemHashHistKernel<DataT, BinnerOp, IdxT, 1>
    <<<blks, ThreadsPerBlock, smemSize, stream>>>(bins, data, nrows, nbins,
                                                  binner, hashSize);
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
void histogramVecLen(HistType type, int* bins, IdxT nbins, const DataT* data,
                     IdxT nrows, IdxT ncols, cudaStream_t stream,
                     BinnerOp binner) {
  CUDA_CHECK(cudaMemsetAsync(bins, 0, ncols * nbins * sizeof(int), stream));
  switch (type) {
    case HistTypeGmem:
      gmemHist<DataT, BinnerOp, IdxT, VecLen>(bins, nbins, data, nrows, ncols,
                                              binner, stream);
      break;
    case HistTypeSmem:
      smemHist<DataT, BinnerOp, IdxT, VecLen>(bins, nbins, data, nrows, ncols,
                                              binner, stream);
      break;
    case HistTypeSmemBits16:
      smemBitsHist<DataT, BinnerOp, IdxT, 16, VecLen>(bins, nbins, data, nrows,
                                                      ncols, binner, stream);
      break;
    case HistTypeSmemBits8:
      smemBitsHist<DataT, BinnerOp, IdxT, 8, VecLen>(bins, nbins, data, nrows,
                                                     ncols, binner, stream);
      break;
    case HistTypeSmemBits4:
      smemBitsHist<DataT, BinnerOp, IdxT, 4, VecLen>(bins, nbins, data, nrows,
                                                     ncols, binner, stream);
      break;
    case HistTypeSmemBits2:
      smemBitsHist<DataT, BinnerOp, IdxT, 2, VecLen>(bins, nbins, data, nrows,
                                                     ncols, binner, stream);
      break;
    case HistTypeSmemBits1:
      smemBitsHist<DataT, BinnerOp, IdxT, 1, VecLen>(bins, nbins, data, nrows,
                                                     ncols, binner, stream);
      break;
    case HistTypeSmemHash:
      smemHashHist<DataT, BinnerOp, IdxT, VecLen>(bins, nbins, data, nrows,
                                                  ncols, binner, stream);
      break;
    default:
      ASSERT(false, "histogram: Invalid type passed '%d'!", type);
  };
  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename BinnerOp, typename IdxT>
void histogramImpl(HistType type, int* bins, IdxT nbins, const DataT* data,
                   IdxT nrows, IdxT ncols, cudaStream_t stream,
                   BinnerOp binner) {
  size_t bytes = nrows * sizeof(DataT);
  if (nrows <= 0) return;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    histogramVecLen<DataT, BinnerOp, IdxT, 16 / sizeof(DataT)>(
      type, bins, nbins, data, nrows, ncols, stream, binner);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    histogramVecLen<DataT, BinnerOp, IdxT, 8 / sizeof(DataT)>(
      type, bins, nbins, data, nrows, ncols, stream, binner);
  } else if (4 % sizeof(DataT) == 0 && bytes % 4 == 0) {
    histogramVecLen<DataT, BinnerOp, IdxT, 4 / sizeof(DataT)>(
      type, bins, nbins, data, nrows, ncols, stream, binner);
  } else if (2 % sizeof(DataT) == 0 && bytes % 2 == 0) {
    histogramVecLen<DataT, BinnerOp, IdxT, 2 / sizeof(DataT)>(
      type, bins, nbins, data, nrows, ncols, stream, binner);
  } else {
    histogramVecLen<DataT, BinnerOp, IdxT, 1>(type, bins, nbins, data, nrows,
                                              ncols, stream, binner);
  }
}

template <typename IdxT>
HistType selectBestHistAlgo(IdxT nbins) {
  size_t smem = getSharedMemPerBlock();
  size_t requiredSize = nbins * sizeof(unsigned);
  if (requiredSize <= smem) {
    return HistTypeSmem;
  }
  requiredSize = ceildiv<size_t>(nbins, 2) * sizeof(unsigned);
  if (requiredSize <= smem) {
    return HistTypeSmemBits16;
  }
  requiredSize = ceildiv<size_t>(nbins, 4) * sizeof(unsigned);
  if (requiredSize <= smem) {
    return HistTypeSmemBits8;
  }
  requiredSize = ceildiv<size_t>(nbins, 8) * sizeof(unsigned);
  if (requiredSize <= smem) {
    return HistTypeSmemBits4;
  }
  requiredSize = ceildiv<size_t>(nbins, 16) * sizeof(unsigned);
  if (requiredSize <= smem) {
    return HistTypeSmemBits2;
  }
  requiredSize = ceildiv<size_t>(nbins, 32) * sizeof(unsigned);
  if (requiredSize <= smem) {
    return HistTypeSmemBits1;
  }
  return HistTypeGmem;
}

/**
 * @brief Perform histogram on the input data. It chooses the right load size
 * based on the input data vector length. It also supports large-bin cases
 * using a specialized smem-based hashing technique.
 * @tparam DataT input data type
 * @tparam IdxT data type used to compute indices
 * @tparam BinnerOp takes the input data and computes its bin index
 * @param type histogram implementation type to choose
 * @param bins the output bins (length = ncols * nbins)
 * @param nbins number of bins
 * @param data input data (length = ncols * nrows)
 * @param nrows data array length in each column (or batch)
 * @param ncols number of columsn (or batch size)
 * @param stream cuda stream
 * @param op the operation that computes the bin index of the input data
 *
 * @note signature of BinnerOp is `int func(DataT, IdxT);`
 */
template <typename DataT, typename IdxT = int,
          typename BinnerOp = IdentityBinner<DataT, IdxT>>
void histogram(HistType type, int* bins, IdxT nbins, const DataT* data,
               IdxT nrows, IdxT ncols, cudaStream_t stream,
               BinnerOp binner = IdentityBinner<DataT, IdxT>()) {
  HistType computedType = type;
  if (type == HistTypeAuto) {
    computedType = selectBestHistAlgo(nbins);
  }
  histogramImpl<DataT, BinnerOp, IdxT>(computedType, bins, nbins, data, nrows,
                                       ncols, stream, binner);
}

};  // end namespace Stats
};  // end namespace MLCommon
