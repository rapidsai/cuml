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
#include "linalg/custom_accum.h"
#include "linalg/gemm.h"
#include "linalg/row_gemm.h"

namespace MLCommon {
namespace Distance {

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_>
void l1Impl(int m, int n, int k,
            IType const* pA,
            IType const* pB,
            OType const* pC,
            OType* pD,
            OType alpha,
            OType beta,
            cudaStream_t stream=0)
{
  typedef cutlass::Shape<8, 8, 8> AccumulatorsPerThread_t;
  typedef cutlass::Shape<1, 4, 8> ThreadsPerWarp_t;
  typedef cutlass::gemm::LinearScaling<OType> EpilogueFunctor_t;
  typedef LinAlg::ThreadL1NormAdd<AccumulatorsPerThread_t,
          ThreadsPerWarp_t, IType, IType, AccType> MainLoopFunctor_t;

  LinAlg::row_gemm<IType, AccType, OType,
    OutputTile_, AccumulatorsPerThread_t, MainLoopFunctor_t, EpilogueFunctor_t>
    (CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, pA, pB, beta, pC, pD);
}

}
}
