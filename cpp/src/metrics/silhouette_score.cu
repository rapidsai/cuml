
/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include <cuvs/stats/silhouette_score.hpp>

namespace ML {

namespace Metrics {
double silhouette_score(const raft::handle_t& handle,
                        double* y,
                        int nRows,
                        int nCols,
                        int* labels,
                        int nLabels,
                        double* silScores,
                        ML::distance::DistanceType metric)
{
  std::optional<raft::device_vector_view<double, int64_t>> silhouette_score_per_sample;
  if (silScores != NULL) {
    silhouette_score_per_sample = raft::make_device_vector_view<double, int64_t>(silScores, nRows);
  }

  return cuvs::stats::silhouette_score(
    handle,
    raft::make_device_matrix_view<const double, int64_t>(y, nRows, nCols),
    raft::make_device_vector_view<const int, int64_t>(labels, nRows),
    silhouette_score_per_sample,
    nLabels,
    static_cast<cuvs::distance::DistanceType>(metric));
}
}  // namespace Metrics
}  // namespace ML
