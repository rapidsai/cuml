/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "distance.h"

namespace MLCommon {
namespace Distance {

/**
 * @defgroup EpsNeigh Epsilon Neighborhood comptuation
 * @{
 * @brief Constructs an epsilon neighborhood adjacency matrix by filtering the
 *        final distance by some epsilon.
 *
 * @tparam distanceType distance metric to compute between a and b matrices
 * @tparam T            the type of input matrices a and b
 * @tparam Lambda       Lambda function
 * @tparam Index_       Index type
 * @tparam OutputTile_  output tile size per thread
 *
 * @param a         first matrix [row-major] [on device] [dim = m x k]
 * @param b         second matrix [row-major] [on device] [dim = n x k]
 * @param adj       a boolean output adjacency matrix [row-major] [on device]
 *                  [dim = m x n]
 * @param m         number of points in a
 * @param n         number of points in b
 * @param k         dimensionality
 * @param eps       epsilon value to use as a filter for neighborhood
 *                  construction. It is important to note that if the distance
 *                  type returns a squared variant for efficiency, epsilon will
 *                  need to be squared as well.
 * @param workspace temporary workspace needed for computations
 * @param worksize  number of bytes of the workspace
 * @param stream    cuda stream
 * @param fused_op  optional functor taking the output index into c
 *                  and a boolean denoting whether or not the inputs are part of
 *                  the epsilon neighborhood.
 * @return          the workspace size in bytes
 */
template <DistanceType distanceType, typename T, typename Lambda,
          typename Index_ = int, typename OutputTile_ = OutputTile_8x128x128>
size_t epsilon_neighborhood(const T *a, const T *b, bool *adj, Index_ m,
                            Index_ n, Index_ k, T eps, void *workspace,
                            size_t worksize, cudaStream_t stream,
                            Lambda fused_op) {
  auto epsilon_op = [n, eps, fused_op] __device__(T val, Index_ global_c_idx) {
    bool acc = val <= eps;
    fused_op(global_c_idx, acc);
    return acc;
  };

  distance<distanceType, T, T, bool, OutputTile_, decltype(epsilon_op), Index_>(
    a, b, adj, m, n, k, (void *)workspace, worksize, epsilon_op, stream);

  return worksize;
}

template <DistanceType distanceType, typename T, typename Index_ = int,
          typename OutputTile_ = OutputTile_8x128x128>
size_t epsilon_neighborhood(const T *a, const T *b, bool *adj, Index_ m,
                            Index_ n, Index_ k, T eps, void *workspace,
                            size_t worksize, cudaStream_t stream) {
  auto lambda = [] __device__(Index_ c_idx, bool acc) {};
  return epsilon_neighborhood<distanceType, T, decltype(lambda), Index_,
                              OutputTile_>(a, b, adj, m, n, k, eps, workspace,
                                           worksize, stream, lambda);
}
/** @} */

}  // namespace Distance
}  // namespace MLCommon
