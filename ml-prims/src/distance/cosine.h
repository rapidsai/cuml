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
#include "linalg/eltwise2d.h"

namespace MLCommon {
namespace Distance {

/**
 * @brief the expanded cosine distance matrix calculation
 *  It computes the following equation: C = op(A^2 + B^2 - 2AB)
 * @tparam IType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam FinalLambda user-defined epilogue lamba
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @tparam in_params user-defined input parameter
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
void cosineAlgo1(int m, int n, int k, InType const *pA, InType const *pB,
                 OutType *pD, AccType *workspace, size_t worksize,
                 FinalLambda fin_op, cudaStream_t stream = 0) {
  typedef ExpandedDistanceFragmentMultiplyAdd<CosFusedDistance>
    FragmentMultiplyAdd_;
  auto norm_op = [] __device__(AccType in) { return mySqrt(in); };
  distanceAlgo1<InType, AccType, OutType, OutputTile_, FragmentMultiplyAdd_>(
    m, n, k, pA, pB, pD, false, workspace, worksize, fin_op, norm_op, stream);
}

}; // end namespace Distance
}; // end namespace MLCommon
