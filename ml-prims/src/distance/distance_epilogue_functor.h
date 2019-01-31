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

template <typename InputScalar_,
          typename OutputScalar_,
          typename InParams_,
          typename OutParams_,
          typename GemmConfig_,
          typename FragmentMultiplyAdd_ =
              DistanceFragmentMultiplyAdd<OutputScalar_,
                                        InParams_,
                                        OutParams_> >
struct DistanceEpilogueFunctor {
  // The scalar.
  typedef InputScalar_ InputScalar;
  typedef OutputScalar_ Scalar;

  // The custom params
  typedef InParams_ InParams;
  typedef OutParams_ OutParams;

  // The adapater.
  typedef FragmentMultiplyAdd_ FragmentMultiplyAdd;

  /// The number of iterations in the epilogue.
  /// TODO(minseok): GemmEpilogueTraitsHelper's duplicate
  typedef cutlass::Shape<1,
                         GemmConfig_::MultiplyAdd::AccumulatorsPerThread::kH /
                             GemmConfig_::kAccumulatorsPerLdsB,
                         GemmConfig_::kAccumulatorsPerLdsB>
      Iterations;

  /// The iteration strides in the H/W dimension.
  /// TODO(minseok): GemmEpilogueTraitsHelper's duplicate
  typedef cutlass::Shape<0,
                         GemmConfig_::kAccumulatorsPerLdsB*(
                            GemmConfig_::Warps::kH* GemmConfig_::MultiplyAdd::ThreadsPerWarp::kH - 1),
                         0>
      Delta;

  /// The traits class to build the iterator to load data from global memory for AA
  typedef DistanceGlobalTileAATraits<
      InputScalar const,
      cutlass::Shape<1,
            GemmConfig_::OutputTile::kH / cutlass::ShapeCount<Iterations>::kCount,
            GemmConfig_::OutputTile::kW>,
      cutlass::Shape<1,
            cutlass::ShapeCount<typename GemmConfig_::Warps>::kCount,
            GemmConfig_::kWarpSize>,
      // How many elements do we jump over at each iteration?
      Iterations::kW,
      GemmConfig_::kScalarsPerLdgA>
      GlobalLoadTileAATraits;
  /// The iterator for AA in global memory.
  typedef cutlass::gemm::GemmGlobalIteratorCd<GlobalLoadTileAATraits, int> GlobalLoadIteratorAA;

  /// The traits class to build the iterator to load data from global memory for BB
  typedef DistanceGlobalTileBBTraits<
      InputScalar const,
      cutlass::Shape<1,
            GemmConfig_::OutputTile::kH / cutlass::ShapeCount<Iterations>::kCount,
            GemmConfig_::OutputTile::kW>,
      cutlass::Shape<1,
            cutlass::ShapeCount<typename GemmConfig_::Warps>::kCount,
            GemmConfig_::kWarpSize>,
      GemmConfig_::kScalarsPerLdgB>
      GlobalLoadTileBBTraits;
  /// The iterator for BB in global memory.
  typedef cutlass::gemm::GemmGlobalIteratorCd<GlobalLoadTileBBTraits, int> GlobalLoadIteratorBB;
  
  /// The parameters.
  struct Params {
    /// The alpha/beta scaling params.
    Scalar alpha, beta;

    /// sqrt on/off
    bool enable_sqrt;

    /// The input params for customizing epilogues
    InParams in_params;
    /// The output params for customizing epilogues
    OutParams out_params;
 
    /// The params for the aa column vector iterator.
    typename GlobalLoadIteratorAA::Params iterator_aa;
    /// The params for the bb row vector iterator.
    typename GlobalLoadIteratorBB::Params iterator_bb;

    /// The information from desc
    int m, n, k, ldd;

    /// Initialize the parameters.
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {
      alpha = desc.alpha;
      beta = desc.beta;
      /// row_gemm is assumed
      m = desc.m;
      n = desc.n;
      k = desc.k;
      ldd = desc.ldd;
      return 0;
    }

    /// Initialize the custom parameters. Use code must call it.
    CUTLASS_HOST_DEVICE int initialize(InParams const& in_params, OutParams& out_params) {
      int error_code = 0;
      this->in_params = in_params;
      this->out_params = out_params;

      this->enable_sqrt = in_params.enable_sqrt;

      error_code = iterator_aa.initialize(
          reinterpret_cast<InputScalar const*>(in_params.col_vec),
          1, n, 0, 0);
      if (error_code) {
        return error_code;
      }

      error_code = iterator_bb.initialize(
          reinterpret_cast<InputScalar const*>(in_params.row_vec),
          ldd, 1, 0, 0);
      if (error_code) {
        return error_code;
      }

      return error_code;
    }
  };

  /// Ctor.
  CUTLASS_DEVICE DistanceEpilogueFunctor(Params const& params) : 
      alpha(params.alpha), beta(params.beta),
      in_params(params.in_params),
      out_params(params.out_params),
      fin_op(*(params.in_params.fin_op)){}

  /// Evaluate the functor.
  template <bool enable_sqrt,
            typename FragmentA_,
            typename FragmentB_,
            typename FragmentCol_,
            typename FragmentRow_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum,
                               FragmentB_& output,
                               const int index[FragmentB_::kElements],
                               FragmentCol_ const& col,
                               FragmentRow_ const& row) {
    FragmentMultiplyAdd mad;
    mad.multiply<enable_sqrt>(alpha, accum, output, index, col, row, in_params, out_params, fin_op);
  }

  /// Evaluate the functor.
  template <bool enable_sqrt,
            typename FragmentA_,
            typename FragmentB_,
            typename FragmentCol_,
            typename FragmentRow_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum,
                               FragmentB_ const& old,
                               FragmentB_& output,
                               const int index[FragmentB_::kElements],
                               FragmentCol_ const& col,
                               FragmentRow_ const& row) {
    FragmentMultiplyAdd mad;
    FragmentB_ tmp;
    mad.multiply<enable_sqrt>(beta, old, tmp, index, col, row, in_params, out_params, fin_op);
    mad.multiply_add<enable_sqrt>(alpha, accum, tmp, output, index, col, row, in_params, out_params, fin_op);
  }

  /// The alpha/beta scaling factors.
  Scalar alpha, beta;

  /// The input/output params for customizing epilogues
  InParams in_params;
  OutParams out_params;

  typename InParams::FinalLambda fin_op;
};

} // end namespace Distance
} // end namespace MLCommon

