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
#include <linalg/custom_accum.h>
#include <linalg/cutlass_gemm.cuh>
#include <linalg/eltwise2d.cuh>
#include "algo1.cuh"
#include "distance_fragment_multiply_add.cuh"

#include <cutlass/shape.h>
#include <type_traits>

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
 * @tparam Index_ index type
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
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_ = int>
void euclideanAlgo1(Index_ m, Index_ n, Index_ k, const InType *pA,
                    const InType *pB, OutType *pD, bool enable_sqrt,
                    AccType *workspace, size_t &worksize, FinalLambda fin_op,
                    cudaStream_t stream, bool isRowMajor) {
  typedef ExpandedDistanceFragmentMultiplyAdd<L2FusedDistance>
    FragmentMultiplyAdd_;
  auto norm_op = [] __device__(InType in) { return in; };
  distanceAlgo1<InType, AccType, OutType, OutputTile_, FragmentMultiplyAdd_,
                FinalLambda, decltype(norm_op), Index_>(
    m, n, k, pA, pB, pD, enable_sqrt, workspace, worksize, fin_op, norm_op,
    stream, isRowMajor);
}

/**
 * @brief the unexpanded euclidean distance matrix calculation
 *  It computes the following equation: cij = op((ai-bj)^2)
 * @tparam InType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OutType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam Index_ index type
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @param enable_sqrt if the square root is computed or not
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream where to launch work
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_ = int>
void euclideanAlgo2(Index_ m, Index_ n, Index_ k, const InType *pA,
                    const InType *pB, OutType *pD, bool enable_sqrt,
                    FinalLambda fin_op, cudaStream_t stream, bool isRowMajor) {
  typedef std::is_same<OutType, bool> is_bool;
  typedef typename std::conditional<is_bool::value, AccType, OutType>::type
    EffOutType;
  EffOutType *pDCast =
    reinterpret_cast<EffOutType *>(pD);  // Pretend to be EffOutType;

  typedef cutlass::Shape<8, 8, 8> AccumulatorsPerThread_;
  typedef LinAlg::ThreadDiffSquaredAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, InType, InType, AccType>
    MainLoopFunctor_;
  typedef LinAlg::CustomGemmConfig<InType, AccType, EffOutType, OutputTile_,
                                   AccumulatorsPerThread_, MainLoopFunctor_>
    GemmConfig_;

  typedef UnexpandedDistanceFragmentMultiplyAdd FragmentMultiplyAdd_;

  typedef UnexpandedDistanceEpilogueFunctor<EffOutType, GemmConfig_,
                                            FragmentMultiplyAdd_>
    EpilogueFunctor_;

  typedef typename std::conditional<
    is_bool::value,
    BoolEpilogueTraitsHelper<GemmConfig_, EpilogueFunctor_, Index_>,
    cutlass::gemm::GemmEpilogueTraitsHelper<
      GemmConfig_, EpilogueFunctor_, Index_>>::type EpilogueTraitsHelper_;

  typedef typename cutlass::gemm::SimplifiedGemmEpilogueTraits<
    GemmConfig_, EpilogueFunctor_, Index_, EpilogueTraitsHelper_>
    GemmEpilogueTraits_;
  typedef UnexpandedDistanceGemmEpilogue<GemmEpilogueTraits_> GemmEpilogue_;
  typedef typename EpilogueFunctor_::Params EpiParams;

  cublasOperation_t transa, transb;
  const InType *aPtr, *bPtr;
  Index_ lda, ldb, ldd;
  Index_ gemm_m, gemm_n;
  if (isRowMajor) {
    transa = CUBLAS_OP_T;
    transb = CUBLAS_OP_N;
    aPtr = pB;
    bPtr = pA;
    lda = ldb = k;
    ldd = n;
    gemm_m = n;
    gemm_n = m;
  } else {
    transa = CUBLAS_OP_N;
    transb = CUBLAS_OP_T;
    aPtr = pA;
    bPtr = pB;
    lda = m;
    ldb = n;
    ldd = m;
    gemm_m = m;
    gemm_n = n;
  }
  LinAlg::gemm<InType, AccType, EffOutType, OutputTile_, AccumulatorsPerThread_,
               MainLoopFunctor_, Index_, GemmConfig_, EpilogueFunctor_,
               GemmEpilogueTraits_, GemmEpilogue_>(
    transa, transb, gemm_m, gemm_n, k, (EffOutType)1, aPtr, lda, bPtr, ldb,
    (EffOutType)0, nullptr, ldd, pDCast,
    [enable_sqrt] HD(EpiParams & p) {
      int err = p.initializeExtra(nullptr, nullptr, enable_sqrt);
      return err;
    },
    fin_op, stream);
}

};  // end namespace Distance
};  // end namespace MLCommon
