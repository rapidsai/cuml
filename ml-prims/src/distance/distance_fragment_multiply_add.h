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

#include <cutlass/fragment.h>
#include <cutlass/shape.h>

namespace MLCommon {
namespace Distance {

template <typename Scalar_,
          typename InParams_,
          typename OutParams_>
struct DistanceFragmentMultiplyAdd {
  /// The shape of the instruction.
  typedef cutlass::Shape<1, 1, 1, 1> InstructionShape;
  /// The type for A.
  typedef Scalar_ ScalarA;
  /// The type for B.
  typedef Scalar_ ScalarB;
  /// The type for C.
  typedef Scalar_ ScalarC;
  /// The type for D.
  typedef Scalar_ ScalarD;

  /// Ctor.
  CUTLASS_DEVICE DistanceFragmentMultiplyAdd() {}

  /// Multiply : d = a*b.
  template <bool enable_sqrt_,
            typename FragmentB_,
            typename FragmentCd_,
            typename FragmentCol_,
            typename FragmentRow_,
            typename Lambda_>
  CUTLASS_DEVICE void multiply(Scalar_ a,
                               FragmentB_ const& b,
                               FragmentCd_& d,
                               const int index[FragmentCd_::kElements],
                               FragmentCol_ const& col,
                               FragmentRow_ const& row,
                               InParams_ const& in_params,
                               OutParams_& out_params,
                               Lambda_ fin_op) {
    int const kReduction = FragmentB_::kElements / FragmentCd_::kElements;
    int const width =
        FragmentCd_::kElements / FragmentCol_::kElements;
    for (int j = 0; j < FragmentCd_::kElements; ++j) {
      auto accum = a * b[j * kReduction + 0];
      for (int k = 1; k < kReduction; ++k) {
        accum += a * b[j * kReduction + k];
      }
      accum = col[j / width] + row[j % width] - 2 *accum;
      if(enable_sqrt_)
        accum = sqrt(accum);
      d[j] = (index[j] == -1)? accum : fin_op(accum, index[j], in_params, out_params);
    }
  }

  /// Multiply : d = a*b + c.
  template <bool enable_sqrt_,
            typename FragmentB_,
            typename FragmentCd_,
            typename FragmentCol_,
            typename FragmentRow_,
            typename Lambda_>
  CUTLASS_DEVICE void multiply_add(Scalar_ a,
                                   FragmentB_ const& b,
                                   FragmentCd_ const& c,
                                   FragmentCd_& d,
                                   const int index[FragmentCd_::kElements],
                                   FragmentCol_ const& col,
                                   FragmentRow_ const& row,
                                   InParams_ const& in_params,
                                   OutParams_& out_params,
                                   Lambda_ fin_op) {
    int const kReduction = FragmentB_::kElements / FragmentCd_::kElements;
    int const width =
        FragmentCd_::kElements / FragmentCol_::kElements;
    for (int j = 0; j < FragmentCd_::kElements; ++j) {
      auto accum = a * b[j * kReduction + 0] + c[j];
      for (int k = 1; k < kReduction; ++k) {
        accum += a * b[j * kReduction + k];
      }
      accum = col[j / width] + row[j % width] - 2 * accum;
      if(enable_sqrt_)
        accum = sqrt(accum);
      d[j] = (index[j] == -1)? accum : fin_op(accum, index[j], in_params, out_params);
    }
  }
};

} // end namespace Distance
} // end namespace MLCommon

