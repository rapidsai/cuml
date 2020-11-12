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

#include <linalg/cutlass_gemm.cuh>
#include <raft/linalg/norm.cuh>
#include "distance_epilogue.cuh"
#include "distance_epilogue_functor.cuh"
#include "distance_epilogue_traits.h"
#include "distance_fragment_multiply_add.cuh"

#include <cutlass/gemm/gemm_epilogue_traits.h>
#include <cutlass/gemm/thread_multiply_add.h>
#include <cutlass/shape.h>

#include <type_traits>

namespace MLCommon {
namespace Distance {

/**
 * @brief the expanded distance matrix calculation
 *  It computes the following equation: C = op(A^2 + B^2 - 2AB)
 * @tparam IType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OutType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam FragmentMultiplyAdd_ cutlass-fragment-level multiply & add
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam NormLambda the final L2 norm lambda
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
 * @param fin_op the final element-wise epilogue lambda
 * @param norm_op the final L2 norm lambda
 * @param stream cuda stream where to launch work
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FragmentMultiplyAdd_,
          typename FinalLambda, typename NormLambda, typename Index_ = int>
void distanceAlgo1(Index_ m, Index_ n, Index_ k, const InType *pA,
                   const InType *pB, OutType *pD, bool enable_sqrt,
                   AccType *workspace, size_t worksize, FinalLambda fin_op,
                   NormLambda norm_op, cudaStream_t stream, bool isRowMajor) {
  typedef std::is_same<OutType, bool> is_bool;
  typedef typename std::conditional<is_bool::value, AccType, OutType>::type
    EffOutType;
  EffOutType *pDCast =
    reinterpret_cast<EffOutType *>(pD);  // Pretend to be EffOutType;

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
    raft::linalg::rowNorm(col_vec, pA, k, m, raft::linalg::L2Norm, isRowMajor,
                          stream, norm_op);
    raft::linalg::rowNorm(row_vec, pB, k, n, raft::linalg::L2Norm, isRowMajor,
                          stream, norm_op);
  } else {
    raft::linalg::rowNorm(col_vec, pA, k, m, raft::linalg::L2Norm, isRowMajor,
                          stream, norm_op);
  }

  typedef typename cutlass::Shape<8, 8, 8> AccumulatorsPerThread_;
  typedef cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, InType, InType, AccType>
    MainLoopFunctor_;
  typedef LinAlg::CustomGemmConfig<InType, AccType, EffOutType, OutputTile_,
                                   AccumulatorsPerThread_, MainLoopFunctor_>
    GemmConfig_;

  typedef ExpandedDistanceEpilogueFunctor<InType, AccType, GemmConfig_,
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
  typedef ExpandedDistanceGemmEpilogue<GemmEpilogueTraits_> GemmEpilogue_;
  typedef typename EpilogueFunctor_::Params EpiParams;

  cublasOperation_t transa, transb;
  const InType *aPtr, *bPtr;
  Index_ lda, ldb, ldd;
  Index_ gemm_m, gemm_n;
  InType *rvec, *cvec;
  if (isRowMajor) {
    transa = CUBLAS_OP_T;
    transb = CUBLAS_OP_N;
    aPtr = pB;
    bPtr = pA;
    lda = ldb = k;
    ldd = n;
    gemm_m = n;
    gemm_n = m;
    cvec = col_vec;
    rvec = row_vec;
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
    cvec = row_vec;
    rvec = col_vec;
  }
  LinAlg::gemm<InType, AccType, EffOutType, OutputTile_, AccumulatorsPerThread_,
               MainLoopFunctor_, Index_, GemmConfig_, EpilogueFunctor_,
               GemmEpilogueTraits_, GemmEpilogue_>(
    transa, transb, gemm_m, gemm_n, k, (EffOutType)1, aPtr, lda, bPtr, ldb,
    (EffOutType)0, nullptr, ldd, pDCast,
    [cvec, rvec, enable_sqrt] HD(EpiParams & p) {
      int err = p.initializeExtra(cvec, rvec, enable_sqrt);
      return err;
    },
    fin_op, stream);
}

};  // end namespace Distance
};  // end namespace MLCommon
