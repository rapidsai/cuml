/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/strided_reduction.cuh>

namespace MLCommon {
namespace Stats {

/**
 * @brief Compute the row-wise weighted mean of the input matrix
 *
 * @tparam Type the data type
 * @param mu the output mean vector
 * @param data the input matrix (assumed to be row-major)
 * @param weights per-column means
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream to launch work on
 */
template <typename Type>
void rowWeightedMean(Type *mu, const Type *data, const Type *weights, int D,
                     int N, cudaStream_t stream) {
  //sum the weights & copy back to CPU
  Type WS = 0;
  raft::linalg::coalescedReduction(mu, weights, D, 1, (Type)0, stream, false);
  raft::update_host(&WS, mu, 1, stream);

  raft::linalg::coalescedReduction(
    mu, data, D, N, (Type)0, stream, false,
    [weights] __device__(Type v, int i) { return v * weights[i]; },
    [] __device__(Type a, Type b) { return a + b; },
    [WS] __device__(Type v) { return v / WS; });
}

/**
 * @brief Compute the column-wise weighted mean of the input matrix
 *
 * @tparam Type the data type
 * @param mu the output mean vector
 * @param data the input matrix (assumed to be column-major)
 * @param weights per-column means
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream to launch work on
 */
template <typename Type>
void colWeightedMean(Type *mu, const Type *data, const Type *weights, int D,
                     int N, cudaStream_t stream) {
  //sum the weights & copy back to CPU
  Type WS = 0;
  raft::linalg::stridedReduction(mu, weights, 1, N, (Type)0, stream, false);
  raft::update_host(&WS, mu, 1, stream);

  raft::linalg::stridedReduction(
    mu, data, D, N, (Type)0, stream, false,
    [weights] __device__(Type v, int i) { return v * weights[i]; },
    [] __device__(Type a, Type b) { return a + b; },
    [WS] __device__(Type v) { return v / WS; });
}

};  // end namespace Stats
};  // end namespace MLCommon
