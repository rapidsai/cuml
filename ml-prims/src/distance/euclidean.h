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
#include "distance/linear_scaling_sqrt.h"
#include "linalg/custom_accum.h"
#include "linalg/eltwise2d.h"
#include "linalg/gemm.h"
#include "linalg/row_gemm.h"

#include <cutlass/shape.h>

namespace MLCommon {
namespace Distance {

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_,
          typename InParams,
          typename OutParams,
          typename FinalLambda>
void euclideanAlgo1(int m, int n, int k,
                   IType const* pA,
                   IType const* pB,
                   AccType const* pC,
                   OType* pD,
                   OType alpha,
                   OType beta,
                   bool enable_sqrt,
                   InParams const& in_params,
                   OutParams& out_params,
                   AccType* workspace,
                   size_t& worksize,
                   FinalLambda fin_op,
                   cudaStream_t stream=0)
{
  distanceAlgo1<IType, AccType, OType,
                OutputTile_,
                InParams,
                OutParams>(
            m, n, k,
            pA, pB, pC, pD,
            alpha, beta,
            enable_sqrt,
            in_params,
            out_params,
            workspace,
            worksize,
            fin_op,
            stream);
}

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_>
void euclideanAlgo2(int m, int n, int k,
                   IType const* pA,
                   IType const* pB,
                   OType const* pC,
                   OType* pD,
                   OType alpha,
                   OType beta,
                   bool enable_sqrt = false)
{
  typedef cutlass::Shape<8, 8, 8> AccumulatorsPerThread_t;
  typedef cutlass::Shape<1, 4, 8> ThreadsPerWarp_t;
  typedef cutlass::gemm::LinearScaling<OType> EpilogueFunctor_t;
  typedef MLCommon::Distance::LinearScalingSqrt<OType> SqrtEpilogueFunctor_t;
  typedef LinAlg::ThreadDiffSquaredAdd<AccumulatorsPerThread_t,
          ThreadsPerWarp_t, IType, IType, AccType> MainLoopFunctor_t;

  if (enable_sqrt) {
    LinAlg::row_gemm<IType, AccType, OType,
      OutputTile_, AccumulatorsPerThread_t, MainLoopFunctor_t, SqrtEpilogueFunctor_t>
      (CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, pA, pB, beta, pC, pD);
  }
  else {
    LinAlg::row_gemm<IType, AccType, OType,
      OutputTile_, AccumulatorsPerThread_t, MainLoopFunctor_t, EpilogueFunctor_t>
      (CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, pA, pB, beta, pC, pD);
  }
}

}
}
