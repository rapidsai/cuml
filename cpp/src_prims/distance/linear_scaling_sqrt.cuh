/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <cutlass/gemm/linear_scaling.h>
#include "fragment_sqrt.cuh"

namespace MLCommon {
namespace Distance {

template <typename Scalar_,
          typename FragmentMultiplyAdd_ = FragmentSqrt<Scalar_>>
struct LinearScalingSqrt
  : public cutlass::gemm::LinearScaling<Scalar_, FragmentMultiplyAdd_> {
  // Base class
  typedef typename cutlass::gemm::LinearScaling<Scalar_, FragmentMultiplyAdd_>
    Base;
  // The scalar.
  typedef Scalar_ Scalar;
  // The adapater.
  typedef FragmentMultiplyAdd_ FragmentMultiplyAdd;
  /// Ctor.
  ///
  /// @param params The parameters
  ///
  CUTLASS_DEVICE LinearScalingSqrt(typename Base::Params const& params)
    : Base(params) {}

  /// Evaluate the functor.
  ///
  /// @param accum  The accum
  /// @param output The output
  ///
  /// @tparam FragmentA_ { description }
  /// @tparam FragmentB_ { description }
  ///
  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_& output) {
    FragmentMultiplyAdd mad;
    FragmentB_ tmp;
    mad.sqrt(accum, tmp);
    mad.multiply(Base::alpha, tmp, output);
  }

  /// Evaluate the functor.
  ///
  /// @param accum  The accum
  /// @param old    The old
  /// @param output The output
  ///
  /// @tparam FragmentA_ { description }
  /// @tparam FragmentB_ { description }
  ///
  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_ const& old,
                               FragmentB_& output) {
    FragmentMultiplyAdd mad;
    FragmentB_ tmp0, tmp1;
    mad.multiply(Base::beta, old, tmp0);
    mad.sqrt(accum, tmp1);
    mad.multiply_add(Base::alpha, tmp1, tmp0, output);
  }
};

}  // namespace Distance
}  // namespace MLCommon
