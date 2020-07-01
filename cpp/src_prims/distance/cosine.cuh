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
#include <linalg/eltwise2d.cuh>
#include "algo1.cuh"
#include "distance_fragment_multiply_add.cuh"

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
 * @tparam Index_ Index type
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
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_ = int>
void cosineAlgo1(Index_ m, Index_ n, Index_ k, const InType *pA,
                 const InType *pB, OutType *pD, AccType *workspace,
                 size_t worksize, FinalLambda fin_op, cudaStream_t stream,
                 bool isRowMajor) {
  typedef ExpandedDistanceFragmentMultiplyAdd<CosFusedDistance>
    FragmentMultiplyAdd_;
  auto norm_op = [] __device__(AccType in) { return mySqrt(in); };

  // Wrap fin_op to allow computing 1 - pA before calling fin_op
  auto wrapped_fin_op = [fin_op] __device__(AccType d_val, Index_ g_d_idx) {
    return fin_op(static_cast<AccType>(1.0) - d_val, g_d_idx);
  };

  distanceAlgo1<InType, AccType, OutType, OutputTile_, FragmentMultiplyAdd_,
                decltype(wrapped_fin_op), decltype(norm_op), Index_>(
    m, n, k, pA, pB, pD, false, workspace, worksize, wrapped_fin_op, norm_op,
    stream, isRowMajor);
}

};  // end namespace Distance
};  // end namespace MLCommon
