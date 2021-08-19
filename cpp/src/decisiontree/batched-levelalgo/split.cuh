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

#include <limits>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/unary_op.cuh>

namespace ML {
namespace DT {

/**
 * @brief All info pertaining to splitting a node
 *
 * @tparam DataT input data type
 * @tparam std::size_t  indexing type
 */
template <typename DataT>
struct Split {
  typedef Split<DataT> SplitT;

  /** start with this as the initial gain */
  static constexpr DataT Min = -std::numeric_limits<DataT>::max();
  /** special value to represent invalid column id */
  static constexpr std::size_t Invalid = std::numeric_limits<std::size_t>::max();

  /** threshold to compare in this node */
  DataT quesval = Min;
  /** feature index */
  std::size_t colid = 0;
  /** best info gain on this node */
  DataT best_metric_val = Min;
  /** number of samples in the left child */
  std::size_t nLeft = 0;

  DI Split(DataT quesval, std::size_t colid, DataT best_metric_val, std::size_t nLeft)
    : quesval(quesval), colid(colid), best_metric_val(best_metric_val), nLeft(nLeft)
  {
  }
  Split() = default;
  /**
   * @brief Assignment operator overload
   *
   * @param[in] other source object from where to copy
   *
   * @return the reference to the copied object (typically useful for chaining)
   */
  DI SplitT& operator=(const SplitT& other)
  {
    quesval         = other.quesval;
    colid           = other.colid;
    best_metric_val = other.best_metric_val;
    nLeft           = other.nLeft;
    return *this;
  }

  /**
   * @brief updates the current split if the input gain is better
   */
  DI bool update(const SplitT& other)
  {
    bool update_result = false;
    if (other.best_metric_val > best_metric_val) {
      update_result = true;
    } else if (other.best_metric_val == best_metric_val) {
      if (other.colid > colid) {
        update_result = true;
      } else if (other.colid == colid) {
        if (other.quesval > quesval) { update_result = true; }
      }
    }
    if (update_result) { *this = other; }
    return update_result;
  }

  /**
   * @brief reduce the split info in the warp. Best split will be with 0th lane
   */
  DI void warpReduce()
  {
    auto lane = raft::laneId();
#pragma unroll
    for (int i = raft::WarpSize / 2; i >= 1; i /= 2) {
      auto id = lane + i;
      auto qu = raft::shfl(quesval, id);
      auto co = raft::shfl(colid, id);
      auto be = raft::shfl(best_metric_val, id);
      auto nl = raft::shfl(nLeft, id);
      update({qu, co, be, nl});
    }
  }

  /**
   * @brief Computes the best split across the threadblocks
   *
   * @param[in]    smem  shared mem for scratchpad purposes
   * @param[inout] split current split to be updated
   * @param[inout] mutex location which provides exclusive access to node update
   *
   * @note all threads in the block must enter this function together. At the
   *       end thread0 will contain the best split.
   */
  DI void evalBestSplit(void* smem, volatile SplitT* split, int* mutex)
  {
    auto* sbest = reinterpret_cast<SplitT*>(smem);
    warpReduce();
    auto warp   = threadIdx.x / raft::WarpSize;
    auto nWarps = blockDim.x / raft::WarpSize;
    auto lane   = raft::laneId();
    if (lane == 0) sbest[warp] = *this;
    __syncthreads();
    if (warp == 0) {
      if (lane < nWarps)
        *this = sbest[lane];
      else
        *this = SplitT();
      warpReduce();
      // only the first thread will go ahead and update the best split info
      // for current node
      if (threadIdx.x == 0 && this->best_metric_val != Min) {
        while (atomicCAS(mutex, 0, 1))
          ;
        SplitT split_reg;
        split_reg.quesval         = split->quesval;
        split_reg.colid           = split->colid;
        split_reg.best_metric_val = split->best_metric_val;
        split_reg.nLeft           = split->nLeft;
        bool update_result =
          split_reg.update({this->quesval, this->colid, this->best_metric_val, this->nLeft});
        if (update_result) {
          split->quesval         = split_reg.quesval;
          split->colid           = split_reg.colid;
          split->best_metric_val = split_reg.best_metric_val;
          split->nLeft           = split_reg.nLeft;
        }
        __threadfence();
        atomicExch(mutex, 0);
      }
    }
  }
};  // struct Split

/**
 * @brief Initialize the split array
 *
 * @param[out] splits the array to be initialized
 * @param[in]  len    length of this array
 * @param[in]  s      cuda stream where to schedule work
 */
template <typename DataT, int TPB = 256>
void initSplit(Split<DataT>* splits, std::size_t len, cudaStream_t s)
{
  auto op = [] __device__(Split<DataT> * ptr, std::size_t idx) { *ptr = Split<DataT>(); };
  raft::linalg::writeOnlyUnaryOp<Split<DataT>, decltype(op), std::size_t, TPB>(splits, len, op, s);
}

}  // namespace DT
}  // namespace ML
