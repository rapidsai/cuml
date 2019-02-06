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
#include "cuda_utils.h"

namespace MLCommon {
namespace Distance {

/**
 * @brief Fragment-level epilogue function called by ExpandedEpilogueFunctor,
 *  which calls FusedDistance and user lambda
 * @tparam FusedDistance used to generate the final distance value
 */
template <typename FusedDistance>
struct ExpandedDistanceFragmentMultiplyAdd {
  /// Ctor.
  CUTLASS_DEVICE ExpandedDistanceFragmentMultiplyAdd() {}

  /// Multiply : d = b.
  template <bool enable_sqrt_, typename FragmentB_, typename FragmentCd_,
            typename FragmentCol_, typename FragmentRow_, typename Lambda_>
  CUTLASS_DEVICE void multiply(FragmentB_ const &b, FragmentCd_ &d,
                               const int index[FragmentCd_::kElements],
                               FragmentCol_ const &col, FragmentRow_ const &row,
                               Lambda_ fin_op) {
    FusedDistance fd;
    int const kReduction = FragmentB_::kElements / FragmentCd_::kElements;
    int const width = FragmentCd_::kElements / FragmentCol_::kElements;
    for (int j = 0; j < FragmentCd_::kElements; ++j) {
      d[j] = b[j * kReduction + 0];
      for (int k = 1; k < kReduction; ++k) {
        d[j] += b[j * kReduction + k];
      }
      if (index[j] != -1) {
        fd.fused_distance<enable_sqrt_>(d[j], col[j / width], row[j % width]);
        d[j] = fin_op(d[j], index[j]);
      }
    }
  }
};

struct L2FusedDistance {
  /// Ctor.
  CUTLASS_DEVICE L2FusedDistance() {}

  template <bool enable_sqrt_, typename CdElement_, typename ColElement_,
            typename RowElement_>
  CUTLASS_DEVICE void fused_distance(CdElement_ &accum,
                                     ColElement_ const &col_elem,
                                     RowElement_ const &row_elem) {
    accum = col_elem + row_elem - 2 * accum;
    accum = enable_sqrt_ ? mySqrt(accum) : accum;
  }
};

struct CosFusedDistance {
  /// Ctor.
  CUTLASS_DEVICE CosFusedDistance() {}

  template <bool enable_sqrt_, typename CdElement_, typename ColElement_,
            typename RowElement_>
  CUTLASS_DEVICE void fused_distance(CdElement_ &accum,
                                     ColElement_ const &col_elem,
                                     RowElement_ const &row_elem) {
    accum = accum / (col_elem * row_elem);
  }
};

/**
 * @brief Fragment-level epilogue function called by UnexpandedEpilogueFunctor,
 *  which calls the user lambda
 */
struct UnexpandedDistanceFragmentMultiplyAdd {
  /// Ctor.
  CUTLASS_DEVICE UnexpandedDistanceFragmentMultiplyAdd() {}

  /// Multiply : d = b.
  template <bool enable_sqrt_, typename FragmentB_, typename FragmentCd_,
            typename Lambda_>
  CUTLASS_DEVICE void multiply(FragmentB_ const &b, FragmentCd_ &d,
                               const int index[FragmentCd_::kElements],
                               Lambda_ fin_op) {
    int const kReduction = FragmentB_::kElements / FragmentCd_::kElements;
    for (int j = 0; j < FragmentCd_::kElements; ++j) {
      d[j] = b[j * kReduction + 0];
      for (int k = 1; k < kReduction; ++k) {
        d[j] += b[j * kReduction + k];
      }
      if (index[j] != -1) {
        d[j] = enable_sqrt_ ? mySqrt(d[j]) : d[j];
        d[j] = fin_op(d[j], index[j]);
      }
    }
  }
};

} // end namespace Distance
} // end namespace MLCommon
