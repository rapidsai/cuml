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

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_,
          typename AccumulatorsPerThread_ = cutlass::Shape<8,8,8>,
          typename MainLoopFunctor_ =
              cutlass::gemm::ThreadMultiplyAdd<AccumulatorsPerThread_,
                                               cutlass::Shape<1,4,8>,
                                               IType, IType, AccType>, 
          typename InParams_ = NullInParams,
          typename OutParams_ = NullOutParams,
          typename Index_ = int,
          typename GemmConfig_ =
              CustomGemmConfig<IType, OType, OutputTile_,
                               AccumulatorsPerThread_,
                               MainLoopFunctor_>,
          typename EpilogueFunctor_ = cutlass::gemm::LinearScaling<OType>,
          typename GemmEpilogueTraits_ =
              cutlass::gemm::SimplifiedGemmEpilogueTraits<GemmConfig_,
                                                          EpilogueFunctor_,
                                                          Index_>,
          typename GemmEpilogue_ = 
              cutlass::gemm::GemmEpilogue<GemmEpilogueTraits_>,
          typename Lambda>
void row_gemm(cublasOperation_t transA, cublasOperation_t transB,
          int m, int n, int k,
          OType alpha,
          IType const* A, int lda,
          IType const* B, int ldb,
          OType beta,
          OType const* C, int ldc,
          OType* D,
          Lambda op,
          InParams_ const& in_params,
          OutParams_& out_params) {
  gemm<IType, AccType, OType, OutputTile_,
    AccumulatorsPerThread_, MainLoopFunctor_,
    InParams_, OutParams_,
    Index_,
    GemmConfig_,
    EpilogueFunctor_,
    GemmEpilogueTraits_,
    GemmEpilogue_,
    Lambda>
    (transB, transA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, D,
     op, in_params, out_params);

}

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_,
          typename AccumulatorsPerThread_ = cutlass::Shape<8,8,8>,
          typename MainLoopFunctor_ =
              cutlass::gemm::ThreadMultiplyAdd<AccumulatorsPerThread_,
                                               cutlass::Shape<1,4,8>,
                                               IType, IType, AccType>,
          typename EpilogueFunctor_ =
              cutlass::gemm::LinearScaling<OType> >
void row_gemm(cublasOperation_t transA, cublasOperation_t transB,
          int m, int n, int k,
          OType alpha,
          IType const* A, int lda,
          IType const* B, int ldb,
          OType beta,
          OType const* C, int ldc,
          OType* D) {
  gemm<IType, AccType, OType, OutputTile_,
    AccumulatorsPerThread_, MainLoopFunctor_, EpilogueFunctor_>
    (transB, transA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, D);
}

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_,
          typename AccumulatorsPerThread_ = cutlass::Shape<8,8,8>,
          typename MainLoopFunctor_ =
              cutlass::gemm::ThreadMultiplyAdd<AccumulatorsPerThread_,
                                               cutlass::Shape<1,4,8>,
                                               IType, IType, AccType>,
          typename EpilogueFunctor_ =
              cutlass::gemm::LinearScaling<OType> >
void row_gemm(cublasOperation_t transA, cublasOperation_t transB,
          int m, int n, int k,
          OType alpha,
          IType const* A,
          IType const* B,
          OType beta,
          OType const* C,
          OType* D) {
  int lda = (transA == CUBLAS_OP_N) ? k : m;
  int ldb = (transB == CUBLAS_OP_N) ? n : k;
  int ldc = n;  // output is always row-major!
  row_gemm<IType, AccType, OType, OutputTile_,
    AccumulatorsPerThread_, MainLoopFunctor_, EpilogueFunctor_>
    (transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
