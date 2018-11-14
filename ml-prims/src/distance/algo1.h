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

#include "cuda_utils.h"
#include "distance/distance_epilogue_functor.h"
#include "distance/distance_epilogue.h"
#include "distance/distance_fragment_multiply_add.h"
#include "linalg/gemm.h"
#include "linalg/norm.h"
#include "linalg/row_gemm.h"

#include <cutlass/gemm/gemm_epilogue_traits.h>
#include <cutlass/gemm/thread_multiply_add.h>
#include <cutlass/gemm/gemm_epilogue_traits.h>
#include <cutlass/gemm/thread_multiply_add.h>
#include <cutlass/shape.h>


namespace MLCommon {
namespace Distance {

namespace {

template <typename InParams, typename FinalLambda_>
struct InParamsExt : public InParams {
  typedef InParams Base;
  typedef FinalLambda_ FinalLambda;
  InParamsExt(const InParams& in_params_, bool enable_sqrt_,
              void* col_vec_, void* row_vec_, FinalLambda* fin_op_)
      : InParams(in_params_),
        enable_sqrt(enable_sqrt_),
        col_vec(col_vec_),
        row_vec(row_vec_),
        fin_op(fin_op_) {}
  InParamsExt(const InParamsExt& src) = default;
  InParamsExt& operator= (const InParamsExt& src) = default;
  InParamsExt() = default;

  bool enable_sqrt;
  void* col_vec;
  void* row_vec;
  FinalLambda* fin_op; 
};

} // end anonymous namespace

template <typename InType,
          typename AccType,
          typename OutType,
          typename OutputTile_,
          typename InParams,
          typename OutParams,
          typename FinalLambda>
void distanceAlgo1(int m, int n, int k,
                   InType const* pA,
                   InType const* pB,
                   AccType const* pC,
                   OutType* pD,
                   OutType alpha,
                   OutType beta,
                   bool enable_sqrt,
                   InParams const& in_params,
                   OutParams& out_params,
                   AccType* workspace,
                   size_t worksize,
                   FinalLambda fin_op,
                   cudaStream_t stream=0)
{
  if (((pA != pB) && (worksize < (m + n) * sizeof(AccType))) ||
      (worksize < m * sizeof(AccType))) {
    THROW("workspace size error");
  }
  if (workspace == nullptr) {
    THROW("workspace is null");
  }

  typedef InParamsExt<InParams, decltype(fin_op)> InParamsExt_;

  InType* col_vec = workspace;
  InType* row_vec  = workspace;
  if (pA != pB) {
    row_vec += m;
    LinAlg::norm(col_vec, pA, k, m, LinAlg::L2Norm);
    LinAlg::norm(row_vec, pB, k, n, LinAlg::L2Norm);
  }
  else {
    LinAlg::norm(col_vec, pA, k, m, LinAlg::L2Norm);
  }

  InParamsExt_ in_params_ext(in_params, enable_sqrt, col_vec, row_vec, &fin_op);

  typedef typename cutlass::Shape<8, 8, 8> AccumulatorsPerThread_;
  typedef cutlass::gemm::ThreadMultiplyAdd<AccumulatorsPerThread_,
                                           cutlass::Shape<1, 4, 8>,
                                           InType, InType, AccType> MainLoopFunctor_;
  typedef int Index_;
  typedef LinAlg::CustomGemmConfig<InType, OutType,
                                   OutputTile_,
                                   AccumulatorsPerThread_,
                                   MainLoopFunctor_> GemmConfig_;

  typedef DistanceEpilogueFunctor<InType, OutType,
                                  InParamsExt_, OutParams,
                                  GemmConfig_> EpilogueFunctor_;

  typedef typename cutlass::gemm::SimplifiedGemmEpilogueTraits<GemmConfig_,
                                                               EpilogueFunctor_,
                                                               Index_> GemmEpilogueTraits_;
  typedef DistanceGemmEpilogue<GemmEpilogueTraits_> GemmEpilogue_;


  LinAlg::row_gemm<InType, AccType, OutType,
                   OutputTile_,
                   AccumulatorsPerThread_,
                   MainLoopFunctor_,
                   InParamsExt_,
                   OutParams,
                   Index_,
                   GemmConfig_,
                   EpilogueFunctor_,
                   GemmEpilogueTraits_,
                   GemmEpilogue_> 
              (CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k, alpha, pA, k, pB, k, beta, pC, n, pD,
               [] __host__ __device__ (typename EpilogueFunctor_::Params& p,
                   const InParamsExt_& in_params, OutParams& out_params) {
                   int err = p.initialize(in_params, out_params);
                   return err;
               },
               in_params_ext,
               out_params);
}

} // end namespace Distance
} // end namespace MLCommon

