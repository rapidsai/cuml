/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

#include <cuml/common/utils.hpp>

#include <raft/util/cuda_utils.cuh>

#include <stdlib.h>

#include <limits>

namespace MLCommon {
namespace Selection {

/**
 * @brief The comparator
 * @tparam Greater whether to apply greater or lesser than comparison
 * @tparam T data type
 */
template <bool Greater, typename T>
struct Compare {
  /** compare the two input operands */
  static DI bool op(T a, T b) { return Greater ? a > b : a < b; }
};

/**
 * @brief Struct to abstract compare-and-swap operation
 * @tparam TypeV value type
 * @tparam TypeK key type
 */
template <typename TypeV, typename TypeK>
struct KVPair {
  /** the value used to compare and decide for swap */
  TypeV val;
  /** key associated with the value */
  TypeK key;
  typedef KVPair<TypeV, TypeK> Pair;

  /**
   * @brief Compare and swap the current with the other pair
   * @tparam Greater when to perform a swap operation
   * @param other the other pair
   * @param reverse whether the comparison needs to be reversed or not
   */
  template <bool Greater>
  DI void cas(Pair& other, bool reverse)
  {
    bool swap_ = compare<Greater>(other, reverse);
    if (swap_) swap(other);
  }

  /** assign the contents of other pair to the current */
  HDI void operator=(const Pair& other)
  {
    val = other.val;
    key = other.key;
  }

  /** equality comparison */
  DI bool operator==(const Pair& other) const { return val == other.val && key == other.key; }

  /** greater than operator */
  DI bool operator>(const Pair& other) const
  {
    ///@todo: should we also consider the key when values are the same?
    return val > other.val;
  }

  /** lesser than operator */
  DI bool operator<(const Pair& other) const
  {
    ///@todo: should we also consider the key when values are the same?
    return val < other.val;
  }

  /**
   * @brief shuffle the current value with the src laneId
   * @param srcLane the source lane
   * @param width lane width
   * @param mask mask of participating threads (Volta+)
   * @return the shuffled value
   */
  DI Pair shfl(int srcLane, int width = raft::WarpSize, uint32_t mask = 0xffffffffu)
  {
    Pair ret = *this;
    ret.val  = raft::shfl(ret.val, srcLane, width, mask);
    ret.key  = raft::shfl(ret.key, srcLane, width, mask);
    return ret;
  }

  /**
   * @brief XOR-shuffle the current value with the src laneId
   * @param laneMask mask to be applied in order to get the destination lane id
   * @param width lane width
   * @param mask mask of participating threads (Volta+)
   * @return the shuffled value
   */
  DI Pair shfl_xor(int laneMask, int width = raft::WarpSize, uint32_t mask = 0xffffffffu)
  {
    Pair ret = *this;
    ret.val  = raft::shfl_xor(ret.val, laneMask, width, mask);
    ret.key  = raft::shfl_xor(ret.key, laneMask, width, mask);
    return ret;
  }

  /** store the data to global memory */
  DI void store(TypeV* vptr, TypeK* kptr) const
  {
    if (vptr != nullptr) *vptr = val;
    if (kptr != nullptr) *kptr = key;
  }

 private:
  template <bool Greater>
  DI bool compare(const Pair& other, bool reverse)
  {
    return reverse ? Compare<!Greater, TypeV>::op(val, other.val)
                   : Compare<Greater, TypeV>::op(val, other.val);
  }

  DI void swap(Pair& other)
  {
    auto tmp = *this;
    *this    = other;
    other    = tmp;
  }
};

/**
 * @brief perform a warp-wide parallel one-pass bitonic sort stage
 * @tparam TypeV value type
 * @tparam TypeK key type
 * @tparam Greater when to perform swap operation
 * @tparam Log2Stride Starting log2(stride) value
 * @param current current thread's value
 */
template <typename TypeV, typename TypeK, bool Greater, int Log2Stride>
DI void bitonicSortStage(KVPair<TypeV, TypeK>& current)
{
  constexpr int Stride2 = 1 << (Log2Stride + 1);
  int lid               = raft::laneId();
  const bool lidMask    = lid & Stride2;
#pragma unroll
  for (int stage = Log2Stride; stage >= 0; --stage) {
    int stride   = 1 << stage;
    bool group   = lidMask;
    bool phase   = lid & stride;
    bool reverse = phase ^ group;
    auto other   = current.shfl_xor(stride);
    current.cas<Greater>(other, reverse);
  }
}

/**
 * @brief perform a warp-wide parallel bitonic sort
 * @tparam TypeV value type
 * @tparam TypeK key type
 * @tparam Greater when to perform swap operation
 * @param current the pair that needs to be sorted across this warp
 */
template <typename TypeV, typename TypeK, bool Greater>
DI void bitonicSort(KVPair<TypeV, TypeK>& current)
{
  bitonicSortStage<TypeV, TypeK, Greater, 0>(current);
  bitonicSortStage<TypeV, TypeK, Greater, 1>(current);
  bitonicSortStage<TypeV, TypeK, Greater, 2>(current);
  bitonicSortStage<TypeV, TypeK, Greater, 3>(current);
  bitonicSortStage<TypeV, TypeK, Greater, 4>(current);
}

/**
 * @brief perform a warp-wide parallel one-pass bitonic kind of network
 * traversal
 * @tparam TypeV value type
 * @tparam TypeK key type
 * @tparam Greater when to perform swap operation
 * @param current current thread's value
 */
template <typename TypeV, typename TypeK, bool Greater>
DI void warpSort(KVPair<TypeV, TypeK>& current)
{
  int lid = raft::laneId();
#pragma unroll
  for (int stride = raft::WarpSize / 2; stride >= 1; stride /= 2) {
    bool small = !(lid & stride);
    auto other = current.shfl_xor(stride);
    current.cas<Greater>(other, small);
  }
}

/**
 * @brief Struct to abstract an array of key-val pairs.
 * It is assumed to be strided across warp. Meaning, this array is assumed to be
 * actually of length N*32, in row-major order. In other words, all of
 * arr[0] across all threads will come first, followed by arr[1] and so on.
 * @tparam TypeV value type
 * @tparam TypeK key type
 * @tparam N number of elements in the array
 * @tparam Greater whether to do a greater than comparison
 */
template <typename TypeV, typename TypeK, int N, bool Greater>
struct KVArray {
  typedef KVPair<TypeV, TypeK> Pair;
  /** the array of pairs */
  Pair arr[N];
  /** bit-mask representing all valid indices of the array */
  constexpr static int ArrMask = N - 1;
  /** mask representing all threads in a warp */
  constexpr static int WarpMask = raft::WarpSize - 1;

  /** reset the contents of the array */
  DI void reset(TypeV iV, TypeK iK)
  {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      arr[i].val = iV;
      arr[i].key = iK;
    }
  }

  DI void topkUpdate(Pair& other)
  {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      // perform the sort in the reverse order as to minimize the
      // amount of shfl's needed during the merge phase
      bitonicSort<TypeV, TypeK, !Greater>(other);
      arr[i].cas<Greater>(other, false);
      bitonicSort<TypeV, TypeK, Greater>(arr[i]);
    }
  }

  ///@todo: we might just have rewrite this whole thing from scratch!!!
  ///@todo: this fails for N=8 onwards!!
  ///@todo: it also generates "stack frame" for N>=8
  /** sort the elements in this array */
  DI void sort()
  {
    // start by sorting along the warp, first
    warpWideSort();
// iteratively merge each of these "warp-wide" sorted arrays
#pragma unroll
    for (int stride = 1; stride < N; stride *= 2) {
      const int s2 = 2 * stride;
#pragma unroll
      for (int start = 0; start < N; start += s2)
        mergeHalves(stride, start);
#pragma unroll
      for (int start = 0; start < N; start += stride)
        postMergeSort(stride, start);
      warpWideSort();
    }
  }

 private:
  DI void mergeHalves(int stride, int start)
  {
    const int mask = 2 * stride - 1;
#pragma unroll
    for (int i = 0; i < stride; ++i) {
      int src           = i + start;
      int dst           = (i + start) ^ mask;
      auto srcOtherPair = arr[src].shfl_xor(WarpMask);
      auto dstOtherPair = arr[dst].shfl_xor(WarpMask);
      arr[src].cas<Greater>(dstOtherPair, true);
      arr[dst].cas<Greater>(srcOtherPair, false);
    }
  }

  DI void postMergeSort(int stride, int start)
  {
#pragma unroll
    for (int s = stride / 2; s >= 1; s /= 2) {
#pragma unroll
      for (int j = 0; j < s; ++j) {
        int ij = start + j;
        arr[ij].cas<Greater>(arr[ij + s], true);
      }
    }
  }

  DI void warpWideSort()
  {
#pragma unroll
    for (int i = 0; i < N; ++i)
      warpSort<TypeV, TypeK, Greater>(arr[i]);
  }
};

///@todo: specialize this for k=1
template <typename TypeV, typename TypeK, int N, int TPB, bool Greater, bool Sort>
CUML_KERNEL void warpTopKkernel(
  TypeV* outV, TypeK* outK, const TypeV* arr, int k, int rows, int cols, TypeV iV, TypeK iK)
{
  // static_assert(Sort==false, "warpTopK: Sort=true is not yet supported!");

  if (Sort == false) {
    constexpr int RowsPerBlk = TPB / raft::WarpSize;
    const int warpId         = threadIdx.x / raft::WarpSize;
    const int rowId          = blockIdx.x * RowsPerBlk + warpId;
    if (rowId >= rows) return;
    const int maxCols = raft::alignTo(cols, raft::WarpSize);
    KVArray<TypeV, TypeK, N, Greater> topk;
    KVPair<TypeV, TypeK> other;
    topk.reset(iV, iK);
    int colId = threadIdx.x % raft::WarpSize;
    for (; colId < maxCols; colId += raft::WarpSize) {
      auto idx  = rowId * cols + colId;
      other.val = colId < cols ? arr[idx] : iV;
      other.key = colId;
      raft::warpFence();
      topk.topkUpdate(other);
    }
    int lid = raft::laneId();
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int col = i * raft::WarpSize + lid;
      if (outV != nullptr && col < k) outV[rowId * k + col] = topk.arr[i].val;
      if (outK != nullptr && col < k) outK[rowId * k + col] = topk.arr[i].key;
    }  // end for outV and outK
  }  // end for Sort = false
  else {
  }
}

#define CASE_K(kval)                                                       \
  case kval:                                                               \
    warpTopKkernel<TypeV, TypeK, kval, TPB, Greater, Sort>                 \
      <<<nblks, TPB, 0, stream>>>(outV, outK, arr, k, rows, cols, iV, iK); \
    break
/**
 * @brief Perform warp-wide top-k selection on the input matrix
 * @tparam TypeV value type
 * @tparam TypeK key type
 * @tparam Greater whether to do a greater than comparison
 * @tparam Sort whether to sort the final topK values before writing
 * @note the input matrix is assumed to be row-major!
 * @todo verify and extend support to k <= 1024
 */
template <typename TypeV, typename TypeK, bool Greater, bool Sort>
void warpTopK(
  TypeV* outV, TypeK* outK, const TypeV* arr, int k, int rows, TypeK cols, cudaStream_t stream)
{
  static_assert(std::is_same<TypeV, float>::value && (std::is_same<TypeK, int>::value),
                "type not support");
  constexpr int TPB        = 256;
  constexpr int RowsPerBlk = TPB / raft::WarpSize;
  const int nblks          = raft::ceildiv(rows, RowsPerBlk);
  const int kAligned       = raft::alignTo(k, raft::WarpSize) / raft::WarpSize;
  const TypeV iV = Greater ? std::numeric_limits<TypeV>::max() : std::numeric_limits<TypeV>::min();
  const TypeK iK = Greater ? std::numeric_limits<TypeK>::max() : std::numeric_limits<TypeK>::min();
  switch (kAligned) {
    CASE_K(1);
    CASE_K(2);
    CASE_K(3);
    CASE_K(4);
    CASE_K(5);
    CASE_K(6);
    CASE_K(7);
    CASE_K(8);
    CASE_K(9);
    CASE_K(10);
    CASE_K(11);
    CASE_K(12);
    CASE_K(13);
    CASE_K(14);
    CASE_K(15);
    CASE_K(16);
    CASE_K(17);
    CASE_K(18);
    CASE_K(19);
    CASE_K(20);
    CASE_K(21);
    CASE_K(22);
    CASE_K(23);
    CASE_K(24);
    CASE_K(25);
    CASE_K(26);
    CASE_K(27);
    CASE_K(28);
    CASE_K(29);
    CASE_K(30);
    CASE_K(31);
    CASE_K(32);
    default: ASSERT(false, "TopK kernels only support k <= 1024 [%d]", k);
  };
}
#undef CASE_K

};  // end namespace Selection
};  // end namespace MLCommon
