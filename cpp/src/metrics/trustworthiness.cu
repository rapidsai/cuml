/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <distance/distance.cuh>
#include <metrics/scores.cuh>
#include <raft/handle.hpp>

namespace ML {
namespace Metrics {

/**
        * @brief Compute the trustworthiness score
        * @param X[in]: Data in original dimension
        * @param X_embedded[in]: Data in target dimension (embedding)
        * @param n[in]: Number of samples
        * @param m[in]: Number of features in high/original dimension
        * @param d[in]: Number of features in low/embedded dimension
        * @param n_neighbors[in]: Number of neighbors considered by 
        *   trustworthiness score
        * @tparam distance_type: Distance type to consider
        * @return Trustworthiness score
        */
template <typename math_t, raft::distance::DistanceType distance_type>
double trustworthiness_score(const raft::handle_t& h, math_t* X,
                             math_t* X_embedded, int n, int m, int d,
                             int n_neighbors, int batchSize) {
  cudaStream_t stream = h.get_stream();
  auto d_alloc = h.get_device_allocator();

  return MLCommon::Score::trustworthiness_score<math_t, distance_type>(
    X, X_embedded, n, m, d, n_neighbors, d_alloc, stream, batchSize);
}

template double
trustworthiness_score<float, raft::distance::DistanceType::L2SqrtUnexpanded>(
  const raft::handle_t& h, float* X, float* X_embedded, int n, int m, int d,
  int n_neighbors, int batchSize);

};  //end namespace Metrics
};  //end namespace ML
