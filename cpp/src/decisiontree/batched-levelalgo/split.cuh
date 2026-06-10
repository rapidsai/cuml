/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

namespace ML {
namespace DT {

/**
 * @brief All info pertaining to splitting a node
 *
 * @tparam DataT input data type
 */
template <typename DataT>
struct Split {
  typedef Split<DataT> SplitT;
  using CountT = unsigned long long int;
  static_assert(sizeof(CountT) == 8, "RF split counts must be 64-bit.");

  /** start with this as the initial gain */
  static constexpr DataT Min = -std::numeric_limits<DataT>::max();

  /** threshold to compare in this node */
  DataT quesval;
  /** feature index */
  int colid;
  /** best info gain on this node */
  DataT best_metric_val;
  /** global number of samples in the left child */
  CountT global_nLeft;
  /** rank-local number of samples in the left child */
  CountT local_nLeft;

  DI Split(DataT quesval,
           int colid,
           DataT best_metric_val,
           CountT global_nLeft,
           CountT local_nLeft = CountT{0})
    : quesval(quesval),
      colid(colid),
      best_metric_val(best_metric_val),
      global_nLeft(global_nLeft),
      local_nLeft(local_nLeft)
  {
  }

  DI Split()
  {
    quesval = best_metric_val = Min;
    colid                     = -1;
    global_nLeft              = 0;
    local_nLeft               = 0;
  }

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
    global_nLeft    = other.global_nLeft;
    local_nLeft     = other.local_nLeft;
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
      auto gnl = raft::shfl(global_nLeft, id);
      auto lnl = raft::shfl(local_nLeft, id);
      update({qu, co, be, gnl, lnl});
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
      if (threadIdx.x == 0 && this->colid != -1) {
        while (atomicCAS(mutex, 0, 1))
          ;
        SplitT split_reg;
        split_reg.quesval         = split->quesval;
        split_reg.colid           = split->colid;
        split_reg.best_metric_val = split->best_metric_val;
        split_reg.global_nLeft    = split->global_nLeft;
        split_reg.local_nLeft     = split->local_nLeft;
        bool update_result        = split_reg.update({this->quesval,
                                                       this->colid,
                                                       this->best_metric_val,
                                                       this->global_nLeft,
                                                       this->local_nLeft});
        if (update_result) {
          split->quesval         = split_reg.quesval;
          split->colid           = split_reg.colid;
          split->best_metric_val = split_reg.best_metric_val;
          split->global_nLeft    = split_reg.global_nLeft;
          split->local_nLeft     = split_reg.local_nLeft;
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
template <typename DataT, typename IdxT, int TPB = 256>
void initSplit(Split<DataT>* splits, IdxT len, cudaStream_t s)
{
  auto op = [] __device__(Split<DataT> * ptr, IdxT idx) { *ptr = Split<DataT>(); };
  raft::linalg::writeOnlyUnaryOp<Split<DataT>, decltype(op), IdxT, TPB>(splits, len, op, s);
}

template <typename DataT, typename IdxT, int TPB = 256>
void printSplits(Split<DataT>* splits, IdxT len, cudaStream_t s)
{
  auto op = [] __device__(Split<DataT> * ptr, IdxT idx) {
    printf(
      "quesval = %e, colid = %d, best_metric_val = %e, global_nLeft = %llu, local_nLeft = %llu\n",
      ptr->quesval,
      ptr->colid,
      ptr->best_metric_val,
      ptr->global_nLeft,
      ptr->local_nLeft);
  };
  raft::linalg::writeOnlyUnaryOp<Split<DataT>, decltype(op), IdxT, TPB>(splits, len, op, s);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace DT
}  // namespace ML
