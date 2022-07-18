/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <cutlass/cutlass.h>
#include <cutlass/fragment.h>

namespace MLCommon {
namespace LinAlg {

/// Template performing matrix diff-squared-add operation within a thread
template <typename AccumulatorsPerThread_,
          typename ThreadsPerWarp_,
          typename ScalarA_,
          typename ScalarB_,
          typename ScalarC_>
struct ThreadDiffSquaredAdd {
  /// The shape of the instruction.
  typedef cutlass::Shape<1, 1, 1, 1> InstructionShape;
  /// The number of accumulators per thread.
  typedef AccumulatorsPerThread_ AccumulatorsPerThread;
  /// The number of threads per warp.
  typedef ThreadsPerWarp_ ThreadsPerWarp;
  /// The number of accumulators per warp.
  typedef
    typename cutlass::ShapeMul<AccumulatorsPerThread, ThreadsPerWarp>::Shape AccumulatorsPerWarp;
  /// The type for A.
  typedef ScalarA_ ScalarA;
  /// The fragment for A.
  typedef cutlass::Fragment<ScalarA, AccumulatorsPerThread::kW> FragmentA;
  /// The type for B.
  typedef ScalarB_ ScalarB;
  /// The fragment for B.
  typedef cutlass::Fragment<ScalarB, AccumulatorsPerThread::kH> FragmentB;
  /// The type for C and D.
  typedef ScalarC_ ScalarC;
  /// The accumulators.
  typedef cutlass::Fragment<ScalarC, AccumulatorsPerThread::kH * AccumulatorsPerThread::kW, 16>
    Accumulators;

  /// Ctor.
  CUTLASS_DEVICE ThreadDiffSquaredAdd() {}

  /// Multiply : d = (a-b)^2 + c.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d)
  {
    for (int j = 0; j < AccumulatorsPerThread::kH; ++j) {
      for (int i = 0; i < AccumulatorsPerThread::kW; ++i) {
        auto diff      = a[i] - b[j];
        const auto idx = j * AccumulatorsPerThread::kW + i;
        d[idx]         = diff * diff + c[idx];
      }
    }
  }
};

/// Template performing matrix L1-norm operation within a thread
template <typename AccumulatorsPerThread_,
          typename ThreadsPerWarp_,
          typename ScalarA_,
          typename ScalarB_,
          typename ScalarC_>
struct ThreadL1NormAdd {
  /// The shape of the instruction.
  typedef cutlass::Shape<1, 1, 1, 1> InstructionShape;
  /// The number of accumulators per thread.
  typedef AccumulatorsPerThread_ AccumulatorsPerThread;
  /// The number of threads per warp.
  typedef ThreadsPerWarp_ ThreadsPerWarp;
  /// The number of accumulators per warp.
  typedef
    typename cutlass::ShapeMul<AccumulatorsPerThread, ThreadsPerWarp>::Shape AccumulatorsPerWarp;
  /// The type for A.
  typedef ScalarA_ ScalarA;
  /// The fragment for A.
  typedef cutlass::Fragment<ScalarA, AccumulatorsPerThread::kW> FragmentA;
  /// The type for B.
  typedef ScalarB_ ScalarB;
  /// The fragment for B.
  typedef cutlass::Fragment<ScalarB, AccumulatorsPerThread::kH> FragmentB;
  /// The type for C and D.
  typedef ScalarC_ ScalarC;
  /// The accumulators.
  typedef cutlass::Fragment<ScalarC, AccumulatorsPerThread::kH * AccumulatorsPerThread::kW, 16>
    Accumulators;

  /// Ctor.
  CUTLASS_DEVICE ThreadL1NormAdd() {}

  /// Multiply : d = |a-b| + c.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d)
  {
    for (int j = 0; j < AccumulatorsPerThread::kH; ++j) {
      for (int i = 0; i < AccumulatorsPerThread::kW; ++i) {
        auto diff      = a[i] < b[j] ? b[j] - a[i] : a[i] - b[j];
        const auto idx = j * AccumulatorsPerThread::kW + i;
        d[idx]         = diff + c[idx];
      }
    }
  }
};

};  // end namespace LinAlg
};  // end namespace MLCommon
