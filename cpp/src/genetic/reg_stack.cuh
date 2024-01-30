/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#ifndef CUDA_PRAGMA_UNROLL
#ifdef __CUDA_ARCH__
#define CUDA_PRAGMA_UNROLL _Pragma("unroll")
#else
#define CUDA_PRAGMA_UNROLL
#endif  // __CUDA_ARCH__
#endif  // CUDA_PRAGMA_UNROLL
namespace cuml {
namespace genetic {

/**
 * @brief A fixed capacity stack on device currently used for AST evaluation
 *
 * The idea is to use only the registers to store the elements of the stack,
 * thereby achieving the best performance.
 *
 * @tparam DataT   data type of the stack elements
 * @tparam MaxSize max capacity of the stack
 */
template <typename DataT, int MaxSize>
struct stack {
  explicit HDI stack() : elements_(0)
  {
    CUDA_PRAGMA_UNROLL
    for (int i = 0; i < MaxSize; ++i) {
      regs_[i] = DataT(0);
    }
  }

  /** Checks if the stack is empty */
  HDI bool empty() const { return elements_ == 0; }

  /** Current number of elements in the stack */
  HDI int size() const { return elements_; }

  /** Checks if the number of elements in the stack equal its capacity */
  HDI bool full() const { return elements_ == MaxSize; }

  /**
   * @brief Pushes the input element to the top of the stack
   *
   * @param[in] val input element to be pushed
   *
   * @note If called when the stack is already full, then it is a no-op! To keep
   *       the device-side logic simpler, it has been designed this way. Trying
   *       to push more than `MaxSize` elements leads to all sorts of incorrect
   *       behavior.
   */
  HDI void push(DataT val)
  {
    CUDA_PRAGMA_UNROLL
    for (int i = MaxSize - 1; i >= 0; --i) {
      if (elements_ == i) {
        ++elements_;
        regs_[i] = val;
      }
    }
  }

  /**
   * @brief Lazily pops the top element from the stack
   *
   * @return pops the element and returns it, if already reached bottom, then it
   *         returns zero.
   *
   * @note If called when the stack is already empty, then it just returns a
   *       value of zero! To keep the device-side logic simpler, it has been
   *       designed this way. Trying to pop beyond the bottom of the stack leads
   *       to all sorts of incorrect behavior.
   */
  HDI DataT pop()
  {
    CUDA_PRAGMA_UNROLL
    for (int i = 0; i < MaxSize; ++i) {
      if (elements_ == (i + 1)) {
        elements_--;
        return regs_[i];
      }
    }

    return DataT(0);
  }

 private:
  int elements_;
  DataT regs_[MaxSize];
};  // struct stack

}  // namespace genetic
}  // namespace cuml
