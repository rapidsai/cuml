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
#include "distance/algo1.h"
#include "distance/distance_fragment_multiply_add.h"
#include "linalg/custom_accum.h"
#include "linalg/eltwise2d.h"
#include "linalg/gemm.h"
#include "linalg/row_gemm.h"

#include <cutlass/shape.h>

namespace MLCommon {
namespace Distance {

/**
 * @brief the expanded euclidean distance matrix calculation
 *  It computes the following equation: C = op(A^2 + B^2 - 2AB)
 * @tparam InType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OutType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam FinalLambda the final lambda called by FragmentMultiplyAdd_
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @param enable_sqrt if the square root is computed or not
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
void euclideanAlgo1(int m, int n, int k, InType const *pA, InType const *pB,
                    OutType *pD, bool enable_sqrt, AccType *workspace,
                    size_t &worksize, FinalLambda fin_op,
                    cudaStream_t stream = 0) {
  typedef ExpandedDistanceFragmentMultiplyAdd<L2FusedDistance>
    FragmentMultiplyAdd_;
  auto norm_op = [] __device__(InType in) { return in; };
  distanceAlgo1<InType, AccType, OutType, OutputTile_, FragmentMultiplyAdd_>(
    m, n, k, pA, pB, pD, enable_sqrt, workspace, worksize, fin_op, norm_op,
    stream);
}

/**
 * @brief the unexpanded euclidean distance matrix calculation
 *  It computes the following equation: cij = op((ai-bj)^2)
 * @tparam InType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OutType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam FinalLambda user-defined epilogue lamba
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @param enable_sqrt if the square root is computed or not
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
void euclideanAlgo2(int m, int n, int k, InType const *pA, InType const *pB,
                    OutType *pD, bool enable_sqrt, FinalLambda fin_op,
                    cudaStream_t stream = 0) {
  typedef cutlass::Shape<8, 8, 8> AccumulatorsPerThread_;
  typedef LinAlg::ThreadDiffSquaredAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, InType, InType, AccType>
    MainLoopFunctor_;
  typedef int Index_;
  typedef LinAlg::CustomGemmConfig<InType, AccType, OutType, OutputTile_,
                                   AccumulatorsPerThread_, MainLoopFunctor_>
    GemmConfig_;

  typedef UnexpandedDistanceFragmentMultiplyAdd FragmentMultiplyAdd_;

  typedef UnexpandedDistanceEpilogueFunctor<OutType, GemmConfig_,
                                            FragmentMultiplyAdd_>
    EpilogueFunctor_;

  typedef typename cutlass::gemm::SimplifiedGemmEpilogueTraits<
    GemmConfig_, EpilogueFunctor_, Index_>
    GemmEpilogueTraits_;
  typedef UnexpandedDistanceGemmEpilogue<GemmEpilogueTraits_> GemmEpilogue_;
  typedef typename EpilogueFunctor_::Params EpiParams;

  LinAlg::row_gemm<InType, AccType, OutType, OutputTile_,
                   AccumulatorsPerThread_, MainLoopFunctor_, Index_,
                   GemmConfig_, EpilogueFunctor_, GemmEpilogueTraits_,
                   GemmEpilogue_>(
    CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, (OutType)1, pA, k, pB, k, (OutType)0,
    nullptr, n, pD,
    [enable_sqrt] HD (EpiParams & p) {
      int err = p.initializeExtra(nullptr, nullptr, enable_sqrt);
      return err;
    },
    fin_op, stream);
}

}; // end namespace Distance
}; // end namespace MLCommon
