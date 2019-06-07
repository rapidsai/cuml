/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <cutlass/fragment_multiply_add.h>
#include "cuda_utils.h"

namespace MLCommon {
namespace Distance {

template <typename Scalar_>
struct FragmentSqrt : public cutlass::gemm::FragmentMultiplyAdd<Scalar_> {
  /// Base class
  typedef typename cutlass::gemm::FragmentMultiplyAdd<Scalar_> Base;
  /// Ctor.
  CUTLASS_DEVICE FragmentSqrt() : Base() {}

  /// d = sqrt(b).
  template <typename FragmentB_, typename FragmentCd_>
  CUTLASS_DEVICE void sqrt(FragmentB_ const& b, FragmentCd_& d) {
    int const kReduction = FragmentB_::kElements / FragmentCd_::kElements;
    for (int j = 0; j < FragmentCd_::kElements; ++j) {
      d[j] = MLCommon::mySqrt(b[j * kReduction + 0]);
      for (int k = 1; k < kReduction; ++k) {
        d[j] += MLCommon::mySqrt(b[j * kReduction + k]);
      }
    }
  }
};

}  // namespace Distance
}  // namespace MLCommon
