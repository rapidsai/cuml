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


#include "distance/distance_fragment_multiply_add.h"
#include "distance/distance_tile_traits.h"

#include <cutlass/fragment.h>
#include <cutlass/gemm/gemm_global_tile.h>
#include <cutlass/gemm/gemm_traits.h>
#include <cutlass/shape.h>

namespace MLCommon {
namespace Distance {

/**
 * @brief Base EpilogueFunctor for all distance metrics.
 * @tparam InputScalar_  input scalar type
 * @tparam OutputScalar_ output scalar type
 * @tparam FragmentMultiplyAdd_ fragment-level epilogue function
 */
template <typename InputScalar_, typename OutputScalar_,
          typename GemmConfig_, typename FragmentMultiplyAdd_>
struct DistanceEpilogueFunctor {
  // The input scalar
  typedef InputScalar_ InputScalar;

  /// The output scalar
  typedef OutputScalar_ Scalar;

  /// The adapater
  typedef FragmentMultiplyAdd_ FragmentMultiplyAdd;

  /// The number of iterations in the epilogue.
  typedef cutlass::Shape<1,
                         GemmConfig_::MultiplyAdd::AccumulatorsPerThread::kH /
                           GemmConfig_::kAccumulatorsPerLdsB,
                         GemmConfig_::kAccumulatorsPerLdsB>
    Iterations;

  /// The iteration strides in the H/W dimension.
  typedef cutlass::Shape<
    0,
    GemmConfig_::kAccumulatorsPerLdsB *(
      GemmConfig_::Warps::kH *GemmConfig_::MultiplyAdd::ThreadsPerWarp::kH - 1),
    0>
    Delta;

  /// The traits class to build the iterator to load data from global memory for
  /// AA
  typedef DistanceGlobalTileAATraits<
    InputScalar const,
    cutlass::Shape<1, GemmConfig_::OutputTile::kH /
                        cutlass::ShapeCount<Iterations>::kCount,
                   GemmConfig_::OutputTile::kW>,
    cutlass::Shape<1, cutlass::ShapeCount<typename GemmConfig_::Warps>::kCount,
                   GemmConfig_::kWarpSize>,
    // How many elements do we jump over at each iteration?
    Iterations::kW, GemmConfig_::kScalarsPerLdgA>
    GlobalLoadTileAATraits;
  /// The iterator for AA in global memory.
  typedef cutlass::gemm::GemmGlobalIteratorCd<GlobalLoadTileAATraits, int>
    GlobalLoadIteratorAA;

  /// The traits class to build the iterator to load data from global memory for
  /// BB
  typedef DistanceGlobalTileBBTraits<
    InputScalar const,
    cutlass::Shape<1, GemmConfig_::OutputTile::kH /
                        cutlass::ShapeCount<Iterations>::kCount,
                   GemmConfig_::OutputTile::kW>,
    cutlass::Shape<1, cutlass::ShapeCount<typename GemmConfig_::Warps>::kCount,
                   GemmConfig_::kWarpSize>,
    GemmConfig_::kScalarsPerLdgB>
    GlobalLoadTileBBTraits;
  /// The iterator for BB in global memory.
  typedef cutlass::gemm::GemmGlobalIteratorCd<GlobalLoadTileBBTraits, int>
    GlobalLoadIteratorBB;

  /// The parameters.
  struct Params {
    /// sqrt on/off
    bool enable_sqrt;
    /// The params for the aa column vector iterator.
    typename GlobalLoadIteratorAA::Params iterator_aa;
    /// The params for the bb row vector iterator.
    typename GlobalLoadIteratorBB::Params iterator_bb;
    /// The information from desc
    int m, n, k, ldd;

    /// Initialize the parameters.
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const &desc) {
      /// row_gemm is assumed
      m = desc.m;
      n = desc.n;
      k = desc.k;
      ldd = desc.ldd;
      return 0;
    }

    /// Initialize the custom parameters. User code must call it!
    CUTLASS_HOST_DEVICE int initializeExtra(InputScalar *col_vec,
                                            InputScalar *row_vec,
                                            bool enable_sqrt) {
      this->enable_sqrt = enable_sqrt;
      int error_code = iterator_aa.initialize(col_vec, 1, n, 0, 0);
      if (error_code) return error_code;
      error_code = iterator_bb.initialize(row_vec, ldd, 1, 0, 0);
      return error_code;
    }
  }; // end struct Params

  /// Ctor.
  CUTLASS_DEVICE DistanceEpilogueFunctor(Params const &p) : params(p) {}

  /// params
  Params params;
}; // end struct DistanceEpilogueFunctor


/**
 * @brief EpilogueFunctor to work with expanded cases (eg: expanded L2 metric)
 * @tparam InputScalar_  input scalar type
 * @tparam OutputScalar_ output scalar type
 * @tparam FragmentMultiplyAdd_ fragment-level epilogue function
 */
template <typename InputScalar_, typename OutputScalar_, typename GemmConfig_,
          typename FragmentMultiplyAdd_,
          typename BaseClass = DistanceEpilogueFunctor<
            InputScalar_, OutputScalar_, GemmConfig_, FragmentMultiplyAdd_>>
struct ExpandedDistanceEpilogueFunctor : public BaseClass {
  /// Ctor.
  CUTLASS_DEVICE ExpandedDistanceEpilogueFunctor(
    typename BaseClass::Params const &params): BaseClass(params) {}

  /// Evaluate the functor.
  template <typename FragmentA_, typename FragmentB_,
            typename FragmentCol_, typename FragmentRow_, typename FinalLambda>
  CUTLASS_DEVICE void evaluate(FragmentA_ const &accum, FragmentB_ &output,
                               const int index[FragmentB_::kElements],
                               FragmentCol_ const &col,
                               FragmentRow_ const &row, FinalLambda fin_op) {
    FragmentMultiplyAdd_ mad;
    if(this->params.enable_sqrt) {
      mad.multiply<true>(accum, output, index, col, row, fin_op);
    } else {
      mad.multiply<false>(accum, output, index, col, row, fin_op);
    }
  }
};


/**
 * @brief EpilogueFunctor for L1 and unexpanded L2 distance
 * @tparam Scalar_ input scalar type
 * @tparam FragmentMultiplyAdd_ fragment-level epilogue function
 */
template <typename Scalar_, typename GemmConfig_, typename FragmentMultiplyAdd_,
          typename BaseClass = DistanceEpilogueFunctor<
            Scalar_, Scalar_, GemmConfig_, FragmentMultiplyAdd_>>
struct UnexpandedDistanceEpilogueFunctor : public BaseClass {
  /// Ctor.
  CUTLASS_DEVICE UnexpandedDistanceEpilogueFunctor(
    typename BaseClass::Params const &params): BaseClass(params) {}

  /// Evaluate the functor.
  template <typename FragmentA_, typename FragmentB_, typename FinalLambda>
  CUTLASS_DEVICE void evaluate(FragmentA_ const &accum, FragmentB_ &output,
                               const int index[FragmentB_::kElements],
                               FinalLambda fin_op) {
    FragmentMultiplyAdd_ mad;
    if(this->params.enable_sqrt) {
      mad.multiply<true>(accum, output, index, fin_op);
    } else {
      mad.multiply<false>(accum, output, index, fin_op);
    }
  }
};

} // end namespace Distance
} // end namespace MLCommon
