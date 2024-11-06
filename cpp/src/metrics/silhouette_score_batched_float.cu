
/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/distance/distance_types.hpp>

#include <cuvs/stats/silhouette_score.hpp>

namespace ML {
namespace Metrics {
namespace Batched {

float silhouette_score(const raft::handle_t& handle,
                       float* X,
                       int n_rows,
                       int n_cols,
                       int* y,
                       int n_labels,
                       float* scores,
                       int chunk,
                       raft::distance::DistanceType metric)
{
  std::optional<raft::device_vector_view<float, int64_t>> silhouette_score_per_sample;
  if (scores != NULL) {
    silhouette_score_per_sample = raft::make_device_vector_view<float, int64_t>(scores, n_rows);
  }

  return cuvs::stats::silhouette_score_batched(
    handle,
    raft::make_device_matrix_view<const float, int64_t>(X, n_rows, n_cols),
    raft::make_device_vector_view<const int, int64_t>(y, n_rows),
    silhouette_score_per_sample,
    n_labels,
    chunk,
    static_cast<cuvs::distance::DistanceType>(metric));
}
}  // namespace Batched
}  // namespace Metrics
}  // namespace ML
