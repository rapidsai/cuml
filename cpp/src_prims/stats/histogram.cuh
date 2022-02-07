/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <common/seive.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/vectorized.cuh>
#include <stdint.h>

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
  /** shared mem atomics but with bins to be 1b int's */
  HistTypeSmemBits1 = 1,
  /** shared mem atomics but with bins to be 2b int's */
  HistTypeSmemBits2 = 2,
  /** shared mem atomics but with bins to be 4b int's */
  HistTypeSmemBits4 = 4,
  /** shared mem atomics but with bins to ba 1B int's */
  HistTypeSmemBits8 = 8,
  /** shared mem atomics but with bins to be 2B int's */
  HistTypeSmemBits16 = 16,
  /** use only global atomics */
  HistTypeGmem,
  /** uses shared mem atomics to reduce global traffic */
  HistTypeSmem,
  /**
   * uses shared mem atomics with match_any intrinsic to further reduce shared
   * memory traffic. This can only be enabled on Volta and later architectures.
   * If one tries to enable this for older arch's, it will fall back to
   * `HistTypeSmem`.
   * @note This is to be used only when the input dataset leads to a lot of
   *       repetitions in a given warp, else, this algo can be much slower than
   *       `HistTypeSmem`!
   */
  HistTypeSmemMatchAny,
  /** builds a hashmap of active bins in shared mem */
  HistTypeSmemHash,
  /** decide at runtime the best algo for the given inputs */
  HistTypeAuto
};

static const int ThreadsPerBlock = 256;

template <typename IdxT, int VecLen>
dim3 computeGridDim(IdxT nrows, IdxT ncols, const void* kernel)
{
  int occupancy;
  RAFT_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, ThreadsPerBlock, 0));
  const auto maxBlks = occupancy * raft::getMultiProcessorCount();
  int nblksx         = raft::ceildiv<int>(VecLen ? nrows / VecLen : nrows, ThreadsPerBlock);
  // for cases when there aren't a lot of blocks for computing one histogram
  nblksx = std::min(nblksx, maxBlks);
  return dim3(nblksx, ncols);
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen, typename CoreOp>
DI void histCoreOp(const DataT* data, IdxT nrows, IdxT nbins, BinnerOp binner, CoreOp op, IdxT col)
{
  IdxT offset = col * nrows;
  auto bdim   = IdxT(blockDim.x);
  IdxT tid    = threadIdx.x + bdim * blockIdx.x;
  tid *= VecLen;
  IdxT stride = bdim * gridDim.x * VecLen;
  int nCeil   = raft::alignTo<int>(nrows, stride);
  typedef raft::TxN_t<DataT, VecLen> VecType;
  VecType a;
  for (auto i = tid; i < nCeil; i += stride) {
    if (i < nrows) { a.load(data, offset + i); }
#pragma unroll
    for (int j = 0; j < VecLen; ++j) {
      int binId = binner(a.val.data[j], i + j, col);
      op(binId, i + j, col);
    }
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
__global__ void gmemHistKernel(
  int* bins, const DataT* data, IdxT nrows, IdxT nbins, BinnerOp binner)
{
  auto op = [=] __device__(int binId, IdxT row, IdxT col) {
    if (row >= nrows) return;
    auto binOffset = col * nbins;
#if __CUDA_ARCH__ < 700
    raft::myAtomicAdd(bins + binOffset + binId, 1);
#else
    auto amask  = __activemask();
    auto mask   = __match_any_sync(amask, binId);
    auto leader = __ffs(mask) - 1;
    if (raft::laneId() == leader) { raft::myAtomicAdd(bins + binOffset + binId, __popc(mask)); }
#endif  // __CUDA_ARCH__
  };
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, nrows, nbins, binner, op, blockIdx.y);
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
void gmemHist(int* bins,
              IdxT nbins,
              const DataT* data,
              IdxT nrows,
              IdxT ncols,
              BinnerOp binner,
              cudaStream_t stream)
{
  auto blks = computeGridDim<IdxT, VecLen>(
    nrows, ncols, (const void*)gmemHistKernel<DataT, BinnerOp, IdxT, VecLen>);
  gmemHistKernel<DataT, BinnerOp, IdxT, VecLen>
    <<<blks, ThreadsPerBlock, 0, stream>>>(bins, data, nrows, nbins, binner);
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen, bool UseMatchAny>
__global__ void smemHistKernel(
  int* bins, const DataT* data, IdxT nrows, IdxT nbins, BinnerOp binner)
{
  extern __shared__ unsigned sbins[];
  for (auto i = threadIdx.x; i < nbins; i += blockDim.x) {
    sbins[i] = 0;
  }
  __syncthreads();
  auto op = [=] __device__(int binId, IdxT row, IdxT col) {
    if (row >= nrows) return;
#if __CUDA_ARCH__ < 700
    raft::myAtomicAdd<unsigned int>(sbins + binId, 1);
#else
    if (UseMatchAny) {
      auto amask  = __activemask();
      auto mask   = __match_any_sync(amask, binId);
      auto leader = __ffs(mask) - 1;
      if (raft::laneId() == leader) {
        raft::myAtomicAdd<unsigned int>(sbins + binId, __popc(mask));
      }
    } else {
      raft::myAtomicAdd<unsigned int>(sbins + binId, 1);
    }
#endif  // __CUDA_ARCH__
  };
  IdxT col = blockIdx.y;
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, nrows, nbins, binner, op, col);
  __syncthreads();
  auto binOffset = col * nbins;
  for (auto i = threadIdx.x; i < nbins; i += blockDim.x) {
    auto val = sbins[i];
    if (val > 0) { raft::myAtomicAdd<unsigned int>((unsigned int*)bins + binOffset + i, val); }
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen, bool UseMatchAny>
void smemHist(int* bins,
              IdxT nbins,
              const DataT* data,
              IdxT nrows,
              IdxT ncols,
              BinnerOp binner,
              cudaStream_t stream)
{
  auto blks = computeGridDim<IdxT, VecLen>(
    nrows, ncols, (const void*)smemHistKernel<DataT, BinnerOp, IdxT, VecLen, UseMatchAny>);
  size_t smemSize = nbins * sizeof(unsigned);
  smemHistKernel<DataT, BinnerOp, IdxT, VecLen, UseMatchAny>
    <<<blks, ThreadsPerBlock, smemSize, stream>>>(bins, data, nrows, nbins, binner);
}

template <unsigned _BIN_BITS>
struct BitsInfo {
  static unsigned const BIN_BITS  = _BIN_BITS;
  static unsigned const WORD_BITS = sizeof(unsigned) * 8;
  static unsigned const WORD_BINS = WORD_BITS / BIN_BITS;
  static unsigned const BIN_MASK  = (1 << BIN_BITS) - 1;
};

template <unsigned BIN_BITS>
DI void incrementBin(unsigned* sbins, int* bins, int nbins, int binId)
{
  typedef BitsInfo<BIN_BITS> Bits;
  auto iword    = binId / Bits::WORD_BINS;
  auto ibin     = binId % Bits::WORD_BINS;
  auto sh       = ibin * Bits::BIN_BITS;
  auto old_word = atomicAdd(sbins + iword, unsigned(1 << sh));
  auto new_word = old_word + unsigned(1 << sh);
  if ((new_word >> sh & Bits::BIN_MASK) != 0) return;
  // overflow
  raft::myAtomicAdd<unsigned int>((unsigned int*)bins + binId, Bits::BIN_MASK + 1);
  for (int dbin = 1; ibin + dbin < Bits::WORD_BINS && binId + dbin < nbins; ++dbin) {
    auto sh1 = (ibin + dbin) * Bits::BIN_BITS;
    if ((new_word >> sh1 & Bits::BIN_MASK) == 0) {
      // overflow
      raft::myAtomicAdd<unsigned int>((unsigned int*)bins + binId + dbin, Bits::BIN_MASK);
    } else {
      // correction
      raft::myAtomicAdd(bins + binId + dbin, -1);
      break;
    }
  }
}

template <>
DI void incrementBin<1>(unsigned* sbins, int* bins, int nbins, int binId)
{
  typedef BitsInfo<1> Bits;
  auto iword    = binId / Bits::WORD_BITS;
  auto sh       = binId % Bits::WORD_BITS;
  auto old_word = atomicXor(sbins + iword, unsigned(1 << sh));
  if ((old_word >> sh & 1) != 0) raft::myAtomicAdd(bins + binId, 2);
}

template <typename DataT, typename BinnerOp, typename IdxT, int BIN_BITS, int VecLen>
__global__ void smemBitsHistKernel(
  int* bins, const DataT* data, IdxT nrows, IdxT nbins, BinnerOp binner)
{
  extern __shared__ unsigned sbins[];
  typedef BitsInfo<BIN_BITS> Bits;
  auto nwords = raft::ceildiv<int>(nbins, Bits::WORD_BINS);
  for (auto j = threadIdx.x; j < nwords; j += blockDim.x) {
    sbins[j] = 0;
  }
  __syncthreads();
  IdxT col       = blockIdx.y;
  IdxT binOffset = col * nbins;
  auto op        = [=] __device__(int binId, IdxT row, IdxT col) {
    if (row >= nrows) return;
    incrementBin<Bits::BIN_BITS>(sbins, bins + binOffset, (int)nbins, binId);
  };
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, nrows, nbins, binner, op, col);
  __syncthreads();
  for (auto j = threadIdx.x; j < (int)nbins; j += blockDim.x) {
    auto shift = j % Bits::WORD_BINS * Bits::BIN_BITS;
    int count  = sbins[j / Bits::WORD_BINS] >> shift & Bits::BIN_MASK;
    if (count > 0) raft::myAtomicAdd(bins + binOffset + j, count);
  }
}

template <typename DataT, typename BinnerOp, typename IdxT, int BIN_BITS, int VecLen>
void smemBitsHist(int* bins,
                  IdxT nbins,
                  const DataT* data,
                  IdxT nrows,
                  IdxT ncols,
                  BinnerOp binner,
                  cudaStream_t stream)
{
  typedef BitsInfo<BIN_BITS> Bits;
  auto blks = computeGridDim<IdxT, VecLen>(
    nrows, ncols, (const void*)smemBitsHistKernel<DataT, BinnerOp, IdxT, Bits::BIN_BITS, VecLen>);
  size_t smemSize = raft::ceildiv<size_t>(nbins, Bits::WORD_BITS / Bits::BIN_BITS) * sizeof(int);
  smemBitsHistKernel<DataT, BinnerOp, IdxT, Bits::BIN_BITS, VecLen>
    <<<blks, ThreadsPerBlock, smemSize, stream>>>(bins, data, nrows, nbins, binner);
}

#define INVALID_KEY -1
DI void clearHashTable(int2* ht, int hashSize)
{
  for (auto i = threadIdx.x; i < hashSize; i += blockDim.x) {
    ht[i] = {INVALID_KEY, 0};
  }
}

DI int findEntry(int2* ht, int hashSize, int binId, int threshold)
{
  int idx = binId % hashSize;
  int t;
  int count = 0;
  while ((t = atomicCAS(&(ht[idx].x), INVALID_KEY, binId)) != INVALID_KEY && t != binId) {
    ++count;
    if (count >= threshold) {
      idx = INVALID_KEY;
      break;
    }
    ++idx;
    if (idx >= hashSize) { idx = 0; }
  }
  return idx;
}

DI void flushHashTable(int2* ht, int hashSize, int* bins, int nbins, int col)
{
  int binOffset = col * nbins;
  for (auto i = threadIdx.x; i < hashSize; i += blockDim.x) {
    if (ht[i].x != INVALID_KEY && ht[i].y > 0) {
      raft::myAtomicAdd(bins + binOffset + ht[i].x, ht[i].y);
    }
    ht[i] = {INVALID_KEY, 0};
  }
}
#undef INVALID_KEY

///@todo: honor VecLen template param
template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
__global__ void smemHashHistKernel(int* bins,
                                   const DataT* data,
                                   IdxT nrows,
                                   IdxT nbins,
                                   BinnerOp binner,
                                   int hashSize,
                                   int threshold)
{
  extern __shared__ int2 ht[];
  int* needFlush = (int*)&(ht[hashSize]);
  if (threadIdx.x == 0) { needFlush[0] = 0; }
  clearHashTable(ht, hashSize);
  __syncthreads();
  auto op = [=] __device__(int binId, IdxT row, IdxT col) {
    bool iNeedFlush = false;
    if (row < nrows) {
      int hidx = findEntry(ht, hashSize, binId, threshold);
      if (hidx >= 0) {
        raft::myAtomicAdd(&(ht[hidx].y), 1);
      } else {
        needFlush[0] = 1;
        iNeedFlush   = true;
      }
    }
    __syncthreads();
    if (needFlush[0]) {
      flushHashTable(ht, hashSize, bins, nbins, col);
      __syncthreads();
      if (threadIdx.x == 0) { needFlush[0] = 0; }
      __syncthreads();
    }
    if (iNeedFlush) {
      int hidx = findEntry(ht, hashSize, binId, threshold);
      // all threads are bound to get one valid entry as all threads in this
      // block will make forward progress due to the __syncthreads call in the
      // subsequent iteration
      raft::myAtomicAdd(&(ht[hidx].y), 1);
    }
  };
  IdxT col = blockIdx.y;
  histCoreOp<DataT, BinnerOp, IdxT, VecLen>(data, nrows, nbins, binner, op, col);
  __syncthreads();
  flushHashTable(ht, hashSize, bins, nbins, col);
}

inline int computeHashTableSize()
{
  // we shouldn't have this much of shared memory available anytime soon!
  static const unsigned maxBinsEverPossible = 256 * 1024;
  static Seive primes(maxBinsEverPossible);
  unsigned smem = raft::getSharedMemPerBlock();
  // divide-by-2 because hash table entry stores 2 elements: idx and count
  auto binsPossible = smem / sizeof(unsigned) / 2;
  for (; binsPossible > 1; --binsPossible) {
    if (primes.isPrime(binsPossible)) return (int)binsPossible;
  }
  return 1;  // should not happen!
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
void smemHashHist(int* bins,
                  IdxT nbins,
                  const DataT* data,
                  IdxT nrows,
                  IdxT ncols,
                  BinnerOp binner,
                  cudaStream_t stream)
{
  static const int flushThreshold = 10;
  auto blks                       = computeGridDim<IdxT, 1>(
    nrows, ncols, (const void*)smemHashHistKernel<DataT, BinnerOp, IdxT, 1>);
  int hashSize    = computeHashTableSize();
  size_t smemSize = hashSize * sizeof(int2) + sizeof(int);
  smemHashHistKernel<DataT, BinnerOp, IdxT, 1><<<blks, ThreadsPerBlock, smemSize, stream>>>(
    bins, data, nrows, nbins, binner, hashSize, flushThreshold);
}

template <typename DataT, typename BinnerOp, typename IdxT, int VecLen>
void histogramVecLen(HistType type,
                     int* bins,
                     IdxT nbins,
                     const DataT* data,
                     IdxT nrows,
                     IdxT ncols,
                     cudaStream_t stream,
                     BinnerOp binner)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(bins, 0, ncols * nbins * sizeof(int), stream));
  switch (type) {
    case HistTypeGmem:
      gmemHist<DataT, BinnerOp, IdxT, VecLen>(bins, nbins, data, nrows, ncols, binner, stream);
      break;
    case HistTypeSmem:
      smemHist<DataT, BinnerOp, IdxT, VecLen, false>(
        bins, nbins, data, nrows, ncols, binner, stream);
      break;
    case HistTypeSmemMatchAny:
      smemHist<DataT, BinnerOp, IdxT, VecLen, true>(
        bins, nbins, data, nrows, ncols, binner, stream);
      break;
    case HistTypeSmemBits16:
      smemBitsHist<DataT, BinnerOp, IdxT, 16, VecLen>(
        bins, nbins, data, nrows, ncols, binner, stream);
      break;
    case HistTypeSmemBits8:
      smemBitsHist<DataT, BinnerOp, IdxT, 8, VecLen>(
        bins, nbins, data, nrows, ncols, binner, stream);
      break;
    case HistTypeSmemBits4:
      smemBitsHist<DataT, BinnerOp, IdxT, 4, VecLen>(
        bins, nbins, data, nrows, ncols, binner, stream);
      break;
    case HistTypeSmemBits2:
      smemBitsHist<DataT, BinnerOp, IdxT, 2, VecLen>(
        bins, nbins, data, nrows, ncols, binner, stream);
      break;
    case HistTypeSmemBits1:
      smemBitsHist<DataT, BinnerOp, IdxT, 1, VecLen>(
        bins, nbins, data, nrows, ncols, binner, stream);
      break;
    case HistTypeSmemHash:
      smemHashHist<DataT, BinnerOp, IdxT, VecLen>(bins, nbins, data, nrows, ncols, binner, stream);
      break;
    default: ASSERT(false, "histogram: Invalid type passed '%d'!", type);
  };
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename BinnerOp, typename IdxT>
void histogramImpl(HistType type,
                   int* bins,
                   IdxT nbins,
                   const DataT* data,
                   IdxT nrows,
                   IdxT ncols,
                   cudaStream_t stream,
                   BinnerOp binner)
{
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
    histogramVecLen<DataT, BinnerOp, IdxT, 1>(
      type, bins, nbins, data, nrows, ncols, stream, binner);
  }
}

template <typename IdxT>
HistType selectBestHistAlgo(IdxT nbins)
{
  size_t smem         = raft::getSharedMemPerBlock();
  size_t requiredSize = nbins * sizeof(unsigned);
  if (requiredSize <= smem) { return HistTypeSmem; }
  for (int bits = 16; bits >= 1; bits >>= 1) {
    auto nBytesForBins = raft::ceildiv<size_t>(bits * nbins, 8);
    requiredSize       = raft::alignTo<size_t>(nBytesForBins, sizeof(unsigned));
    if (requiredSize <= smem) { return static_cast<HistType>(bits); }
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
 * @param binner the operation that computes the bin index of the input data
 *
 * @note signature of BinnerOp is `int func(DataT, IdxT);`
 */
template <typename DataT, typename IdxT = int, typename BinnerOp = IdentityBinner<DataT, IdxT>>
void histogram(HistType type,
               int* bins,
               IdxT nbins,
               const DataT* data,
               IdxT nrows,
               IdxT ncols,
               cudaStream_t stream,
               BinnerOp binner = IdentityBinner<DataT, IdxT>())
{
  HistType computedType = type;
  if (type == HistTypeAuto) { computedType = selectBestHistAlgo(nbins); }
  histogramImpl<DataT, BinnerOp, IdxT>(
    computedType, bins, nbins, data, nrows, ncols, stream, binner);
}

};  // end namespace Stats
};  // end namespace MLCommon
