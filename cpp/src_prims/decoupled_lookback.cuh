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

#include <raft/util/cuda_utils.cuh>

namespace MLCommon {

/**
 * @brief Abstraction for computing prefix scan using decoupled lookback
 * Refer to the following link for more details about the algo itself:
 * https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
 * @tparam Type The data structure to compute prefix scan upon. This struct
 * should expose the following operations: = and +=
 */
template <typename Type>
struct DecoupledLookBack {
  /** default ctor */
  DI DecoupledLookBack(void* workspace) : flags((Flags*)workspace) {}

  /**
   * @brief Computes workspace needed (in B) for decoupled lookback
   * @param nblks number of blocks to be launched
   */
  static size_t computeWorkspaceSize(int nblks)
  {
    size_t workspaceSize = sizeof(Flags) * nblks;
    return workspaceSize;
  }

  /**
   * @brief main decoupled lookback operator
   * @param sum the summed value for the current thread
   * @return the inclusive prefix sum computed for the current threadblock
   * @note Should be called unconditionally by all threads in the threadblock!
   */
  DI Type operator()(Type sum)
  {
    sumDone(sum);
    auto prefix = predecessorSum();
    communicateDone(prefix, sum);
    return prefix;
  }

 private:
  struct Flags {
    Type sum;
    Type incl_prefix;
    int status;
  };

  Flags* flags;

  DI bool isLast() { return threadIdx.x == blockDim.x - 1; }

  DI void sumDone(Type sum)
  {
    volatile Flags* myFlag = flags + blockIdx.x;
    __syncthreads();
    if (isLast()) myFlag->sum = sum;
    __threadfence();
    // prefix sum update to be done for the first block
    if (isLast() && blockIdx.x == 0) myFlag->incl_prefix = sum;
    // make sure that sum never crosses flag update!
    __threadfence();
    if (isLast()) myFlag->status = blockIdx.x == 0 ? 2 : 1;
    __threadfence();
  }

  DI Type predecessorSum()
  {
    __shared__ char s_buff[sizeof(Type)];
    auto* s_excl_sum = (Type*)s_buff;
    if (isLast()) {
      int bidx      = blockIdx.x - 1;
      Type excl_sum = 0;
      while (bidx >= 0) {
        volatile Flags* others = flags + bidx;
        int status;
        do {
          status = others->status;
          __threadfence();
        } while (status == 0);
        // one of the predecessors has computed their inclusive sum
        if (status == 2) {
          excl_sum += others->incl_prefix;
          __threadfence();
          break;
        }
        // one of the predessors has only computed it's reduction sum
        if (status == 1) excl_sum += others->sum;
        --bidx;
        __threadfence();
      }
      s_excl_sum[0] = excl_sum;
    }
    __syncthreads();
    return s_excl_sum[0];
  }

  DI void communicateDone(Type prefix, Type sum)
  {
    if (blockIdx.x > 0) {
      volatile Flags* myFlag = flags + blockIdx.x;
      __syncthreads();
      // make sure that the sum never crosses flag update!
      if (isLast()) myFlag->incl_prefix = prefix + sum;
      __threadfence();
      if (isLast()) myFlag->status = 2;
      __threadfence();
    }
  }
};  // end struct DecoupledLookBack

};  // end namespace MLCommon
