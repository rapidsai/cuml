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
#include "linalg/gemm.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @brief the row-major gemm function for the cases with detailed epilogue
 * customization
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
void row_gemm(cublasOperation_t transA, cublasOperation_t transB, int m, int n,
              int k, OType alpha, IType const *A, int lda, IType const *B,
              int ldb, OType beta, OType const *C, int ldc, OType *D, Lambda op,
              FinalLambda fin_op, cudaStream_t stream) {
  gemm<IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
       MainLoopFunctor_, Index_, GemmConfig_, EpilogueFunctor_,
       GemmEpilogueTraits_, GemmEpilogue_>(
    transB, transA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, D, op, fin_op,
    stream);
}

/**
 * @brief the row-major gemm function for the cases no or simple customization
 * is needed
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
 */
template <
  typename IType, typename AccType, typename OType, typename OutputTile_,
  typename AccumulatorsPerThread_ = cutlass::Shape<8, 8, 8>,
  typename MainLoopFunctor_ = cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, IType, IType, AccType>,
  typename EpilogueFunctor_ = LinearScaling<OType>>
void row_gemm(cublasOperation_t transA, cublasOperation_t transB, int m, int n,
              int k, OType alpha, IType const *A, int lda, IType const *B,
              int ldb, OType beta, OType const *C, int ldc, OType *D,
              cudaStream_t stream) {
  gemm<IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
       MainLoopFunctor_, EpilogueFunctor_>(
    transB, transA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, D, stream);
}

/**
 * @brief the simplest row-major gemm function without passing leading
 * dimensions
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
 * @param B input matrix
 * @param beta scalar
 * @param C input matrix
 * @param D output matrix
 * @param stream cuda stream where to launch work
 */
template <
  typename IType, typename AccType, typename OType, typename OutputTile_,
  typename AccumulatorsPerThread_ = cutlass::Shape<8, 8, 8>,
  typename MainLoopFunctor_ = cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, IType, IType, AccType>,
  typename EpilogueFunctor_ = cutlass::gemm::LinearScaling<OType>>
void row_gemm(cublasOperation_t transA, cublasOperation_t transB, int m, int n,
              int k, OType alpha, IType const *A, IType const *B, OType beta,
              OType const *C, OType *D, cudaStream_t stream) {
  int lda = (transA == CUBLAS_OP_N) ? k : m;
  int ldb = (transB == CUBLAS_OP_N) ? n : k;
  int ldc = n; // output is always row-major!
  row_gemm<IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
           MainLoopFunctor_, EpilogueFunctor_>(
    transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, stream);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
