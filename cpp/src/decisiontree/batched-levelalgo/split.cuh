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

#include <cuda_utils.h>

namespace ML {
namespace DecisionTree {

/** All info pertaining to splitting a node */
template <typename DataT, typename IdxT>
struct Split {
  /** start with this as the initial gain */
  static constexpr DataT Min = std::numeric_limits<DataT>::min();
  /** special value to represent invalid column id */
  static constexpr IdxT Invalid = static_cast<IdxT>(-1);

  /** threshold to compare in this node */
  DataT quesval;
  /** feature index */
  IdxT colid;
  /** best info gain on this node */
  DataT best_metric_val;
  /** number of samples in the left child */
  IdxT nLeft;

  DI void init() {
    quesval = best_metric_val = Min;
    colid = Invalid;
    nLeft = 0;
  }

  DI Split<DataT, IdxT>& operator=(const Split<DataT, IdxT>& other) {
    quesval = other.quesval;
    colid = other.colid;
    best_metric_val = other.best_metric_val;
    nLeft = other.nLeft;
  }

  /** updates the current split if the input gain is better */
  DI void update(const Split<DataT, IdxT>& other) {
    if (other.best_metric_val > best_metric_val) *this = other;
  }

  /** reduce the split info in the warp. Best split will be with 0th lane */
  DI void warpReduce() {
    auto lane = MLCommon::laneId();
#pragma unroll
    for (int i = MLCommon::WarpSize / 2; i >= 1; i /= 2) {
      auto id = lane + i;
      auto qu = MLCommon::shfl(quesval, id);
      auto co = MLCommon::shfl(colid, id);
      auto be = MLCommon::shfl(best_metric_val, id);
      auto nl = MLCommon::shfl(nLeft, id);
      update(Split<DataT, IdxT>(qu, co, be, nl));
    }
  }
};  // struct Split

template <typename DataT, typename IdxT>
__global__ void initSplitKernel(Split<DataT, IdxT>* splits, IdxT len) {
  IdxT tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < len) splits[tid].init();
}

/**
 * @brief Initialize the split array
 * @param splits the array to be initialized
 * @param len length of this array
 * @param s cuda stream where to schedule work
 */
template <typename DataT, typename IdxT, int TPB = 256>
void initSplit(Split<DataT, IdxT>* splits, IdxT len, cudaStream_t s) {
  initSplitKernel<DataT, IdxT><<<nblks, TPB, 0, s>>>(splits, len);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace DecisionTree
}  // namespace ML
