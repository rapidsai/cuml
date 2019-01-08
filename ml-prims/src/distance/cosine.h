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
#include "linalg/eltwise2d.h"

namespace MLCommon {
namespace Distance {

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_>
void cosineAlgo1(int m, int n, int k,
                 IType const* pA,
                 IType const* pB,
                 OType const* pC,
                 OType* pD,
                 OType alpha,
                 OType beta,
                 AccType* pWorkspace,
                 size_t workspaceSize,
                 cudaStream_t stream=0)
{
  auto op = [] __device__ (OType a, OType b, OType ab) {
    return ab / (sqrt(a) * sqrt(b));
  };

  auto lambda = [=] (int rows, int cols,
      const OType* dotA, const OType* dotB, const OType* pC, OType* pD,
      OType alpha, OType beta,
      cudaStream_t stream) {
    LinAlg::eltwise2D<OType>(m, n, dotA, dotB, pC, pD, alpha, beta,
      op, stream);
  };

  distanceAlgo1<IType, AccType, OType, OutputTile_>(m, n, k,
    pA, pB, pC, pD,
    alpha, beta,
    pWorkspace, workspaceSize,
    lambda, stream);
}

}
}
