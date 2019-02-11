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
#include "distance/distance_epilogue.h"
#include "distance/distance_epilogue_functor.h"
#include "distance/distance_fragment_multiply_add.h"
#include "linalg/gemm.h"
#include "stats/norm.h"
#include "linalg/row_gemm.h"

#include <cutlass/gemm/gemm_epilogue_traits.h>
#include <cutlass/gemm/thread_multiply_add.h>
#include <cutlass/shape.h>


namespace MLCommon {
namespace Distance {

/**
 * @brief the expanded distance matrix calculation
 *  It computes the following equation: C = op(A^2 + B^2 - 2AB)
 * @tparam IType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam FragmentMultiplyAdd_ cutlass-fragment-level multiply & add
 * @tparam FinalLambda user-defined epilogue lamba
 * @param NormLambda the final L2 norm lambda
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @param enable_sqrt if the square root is computed or not
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final element-wise epilogue lambda
 * @param norm_op the final L2 norm lambda
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FragmentMultiplyAdd_,
          typename FinalLambda, typename NormLambda>
void distanceAlgo1(int m, int n, int k, InType const *pA, InType const *pB,
                   OutType *pD, bool enable_sqrt, AccType *workspace,
                   size_t worksize, FinalLambda fin_op, NormLambda norm_op,
                   cudaStream_t stream = 0) {
  if (((pA != pB) && (worksize < (m + n) * sizeof(AccType))) ||
      (worksize < m * sizeof(AccType))) {
    THROW("workspace size error");
  }
  if (workspace == nullptr) {
    THROW("workspace is null");
  }

  InType *col_vec = workspace;
  InType *row_vec = workspace;
  if (pA != pB) {
    row_vec += m;
    Stats::rowNorm(col_vec, pA, k, m, Stats::L2Norm, norm_op, stream);
    Stats::rowNorm(row_vec, pB, k, n, Stats::L2Norm, norm_op, stream);
  } else {
    Stats::rowNorm(col_vec, pA, k, m, Stats::L2Norm, norm_op, stream);
  }

  typedef typename cutlass::Shape<8, 8, 8> AccumulatorsPerThread_;
  typedef cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, InType, InType, AccType>
    MainLoopFunctor_;
  typedef int Index_;
  typedef LinAlg::CustomGemmConfig<InType, AccType, OutType, OutputTile_,
                                   AccumulatorsPerThread_, MainLoopFunctor_>
    GemmConfig_;

  typedef ExpandedDistanceEpilogueFunctor<
    InType, AccType, GemmConfig_, FragmentMultiplyAdd_>
    EpilogueFunctor_;

  typedef typename cutlass::gemm::SimplifiedGemmEpilogueTraits<
    GemmConfig_, EpilogueFunctor_, Index_>
    GemmEpilogueTraits_;
  typedef ExpandedDistanceGemmEpilogue<GemmEpilogueTraits_> GemmEpilogue_;
  typedef typename EpilogueFunctor_::Params EpiParams;

  LinAlg::row_gemm<InType, AccType, OutType, OutputTile_,
                   AccumulatorsPerThread_, MainLoopFunctor_, Index_,
                   GemmConfig_, EpilogueFunctor_, GemmEpilogueTraits_,
                   GemmEpilogue_>(
    CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, (OutType)1, pA, k, pB, k, (OutType)0,
    nullptr, n, pD,
    [col_vec, row_vec, enable_sqrt] HD (EpiParams & p) {
      int err = p.initializeExtra(col_vec, row_vec, enable_sqrt);
      return err;
    },
    fin_op, stream);
}

}; // end namespace Distance
}; // end namespace MLCommon
