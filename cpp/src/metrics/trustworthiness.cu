/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

#include <cuml/common/distance_type.hpp>
#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/stats/trustworthiness_score.hpp>

namespace ML {
namespace Metrics {

/**
 * @brief Compute the trustworthiness score
 *
 * @param h Raft handle
 * @param X Data in original dimension
 * @param X_embedded Data in target dimension (embedding)
 * @param n Number of samples
 * @param m Number of features in high/original dimension
 * @param d Number of features in low/embedded dimension
 * @param n_neighbors Number of neighbors considered by trustworthiness score
 * @param batchSize Batch size
 * @tparam distance_type: Distance type to consider
 * @return Trustworthiness score
 */
template <typename math_t, ML::distance::DistanceType distance_type>
double trustworthiness_score(const raft::handle_t& h,
                             const math_t* X,
                             math_t* X_embedded,
                             int n,
                             int m,
                             int d,
                             int n_neighbors,
                             int batchSize)
{
  return cuvs::stats::trustworthiness_score(
    h,
    raft::make_device_matrix_view<const math_t, int64_t>(X, n, m),
    raft::make_device_matrix_view<const math_t, int64_t>(X_embedded, n, d),
    n_neighbors,
    static_cast<cuvs::distance::DistanceType>(distance_type),
    batchSize);
}

template double trustworthiness_score<float, ML::distance::DistanceType::L2SqrtUnexpanded>(
  const raft::handle_t& h,
  const float* X,
  float* X_embedded,
  int n,
  int m,
  int d,
  int n_neighbors,
  int batchSize);

};  // end namespace Metrics
};  // end namespace ML
