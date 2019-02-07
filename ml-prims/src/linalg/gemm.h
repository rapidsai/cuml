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

#include <cublas_v2.h>
#include "cublas_wrappers.h"
#include "cutlass_wrappers.h"
#include "cuda_utils.h"


namespace MLCommon {
namespace LinAlg {

/**
 * @brief the gemm function for the cases with detailed epilogue customization
 *  It computes the following equation: D = alpha . opA(A) * opB(B) + beta . C
 * @tparam IType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam AccumulatorsPerThread_ number of accumulators per thread
 * @tparam MainLoopFunctor_ custom functor to be used in the main loop
 * @tparam Index_ the type of index
 * @tparam GemmConfig_ the config for the GEMM
 * @tparam EpilogueFunctor_ custom epilogue functor
 * @tparam GemmEpilogueTraits_ epilogue traits class to build the epilogue
 * @tparam GemmEpilogue_ custom epilogue
 * @tparam Lambda lambda to initialize any custom params inside EpilogueFunctor_
 * @tparam FinalLambda Final device lambda to be applied in epilogue
 * @param transA cublas transpose op for A
 * @param transB cublas transpose op for B
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param alpha scalar
 * @param A input matrix
 * @param lda leading dim for A
 * @param B input matrix
 * @param ldb leading dim for B
 * @param beta scalar
 * @param C input matrix
 * @param ldc leading dim for C and D
 * @param D output matrix
 * @param op lambda function to initialize any custom params inside
 * EpilogueFunctor_
 * @param fin_op the final lambda to be run inside the Epilogue. This can help
 * in customizing a given EpilogueFunctor, without having to go through the task
 * of creating another Functor!
 * @param stream cuda stream where to launch work
 */
template <
  typename IType, typename AccType, typename OType, typename OutputTile_,
  typename AccumulatorsPerThread_ = cutlass::Shape<8, 8, 8>,
  typename MainLoopFunctor_ = cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, IType, IType, AccType>,
  typename Index_ = int,
  typename GemmConfig_ = CustomGemmConfig<
    IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
     MainLoopFunctor_>,
  typename EpilogueFunctor_ = LinearScaling<OType>,
  typename GemmEpilogueTraits_ = cutlass::gemm::SimplifiedGemmEpilogueTraits<
    GemmConfig_, EpilogueFunctor_, Index_>,
  typename GemmEpilogue_ = CustomGemmEpilogue<GemmEpilogueTraits_>,
  typename Lambda, typename FinalLambda>
void gemm(cublasOperation_t transA, cublasOperation_t transB, int m, int n,
          int k, OType alpha, IType const *A, int lda, IType const *B, int ldb,
          OType beta, OType const *C, int ldc, OType *D, Lambda op,
          FinalLambda fin_op, cudaStream_t stream = 0) {
  baseGemm<IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
           MainLoopFunctor_, Index_, GemmConfig_,
           EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
    transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, op, fin_op,
    stream);
}

/**
 * @brief the gemm function for the case where no or simple customization is
 * needed
 *  It computes the following equation: D = alpha . opA(A) * opB(B) + beta . C
 * @tparam IType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam AccumulatorsPerThread_ number of accumulators per thread
 * @tparam EpilogueFunctor_ custom epilogue functor
 * @param transA cublas transpose op for A
 * @param transB cublas transpose op for B
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param alpha scalar
 * @param A input matrix
 * @param lda leading dim for A
 * @param B input matrix
 * @param ldb leading dim for B
 * @param beta scalar
 * @param C input matrix
 * @param ldc leading dim for C and D
 * @param D output matrix
 * @param stream cuda stream where to launch work
 * @{
 */
template <
  typename IType, typename AccType, typename OType, typename OutputTile_,
  typename AccumulatorsPerThread_ = cutlass::Shape<8, 8, 8>,
  typename MainLoopFunctor_ = cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, IType, IType, AccType>,
  typename EpilogueFunctor_ = cutlass::gemm::LinearScaling<OType>>
void gemm(cublasOperation_t transA, cublasOperation_t transB, int m, int n,
          int k, OType alpha, IType const *A, int lda, IType const *B, int ldb,
          OType beta, OType const *C, int ldc, OType *D,
          cudaStream_t stream = 0) {
  typedef CustomGemmConfig<IType, AccType, OType, OutputTile_,
                           AccumulatorsPerThread_, MainLoopFunctor_>
      GemmConfig_;
  gemm<IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
       MainLoopFunctor_, int, GemmConfig_, EpilogueFunctor_>(
           transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D,
           [](typename EpilogueFunctor_::Params &p) { return 0; },
           stream);
}

/**
 * @brief the wrapper of cublas gemm function
 *  It computes the following equation: D = alpha . opA(A) * opB(B) + beta . C
 * @tparam math_t the type of input/output matrices
 * @param a input matrix
 * @param n_rows_a number of rows of A
 * @param n_cols_a number of columns of A
 * @param b input matrix
 * @param c output matrix
 * @param n_rows_c number of rows of C
 * @param n_cols_c number of columns of C
 * @param trans_a cublas transpose op for A
 * @param trans_b cublas transpose op for B
 * @param alpha scalar
 * @param beta scalar
 * @param cublas_h cublas handle
 * @{
 */
template <typename math_t>
void gemm(const math_t *a, int n_rows_a, int n_cols_a, const math_t *b,
          math_t *c, int n_rows_c, int n_cols_c, cublasOperation_t trans_a,
          cublasOperation_t trans_b, math_t alpha, math_t beta,
          cublasHandle_t cublas_h) {
  int m = n_rows_c;
  int n = n_cols_c;
  int k = trans_a == CUBLAS_OP_T ? n_rows_a : n_cols_a;
  int lda = trans_a == CUBLAS_OP_T ? k : m;
  int ldb = trans_b == CUBLAS_OP_T ? n : k;
  int ldc = m;
  CUBLAS_CHECK(LinAlg::cublasgemm(cublas_h, trans_a, trans_b, m, n, k, &alpha,
                                  a, lda, b, ldb, &beta, c, ldc));
}

}; // end namespace LinAlg
}; // end namespace MLCommon
