
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/distance_type.hpp>
#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/stats/silhouette_score.hpp>

namespace ML {
namespace Metrics {
namespace Batched {

double silhouette_score(const raft::handle_t& handle,
                        double* X,
                        int n_rows,
                        int n_cols,
                        int* y,
                        int n_labels,
                        double* scores,
                        int chunk,
                        ML::distance::DistanceType metric)
{
  std::optional<raft::device_vector_view<double, int64_t>> silhouette_score_per_sample;
  if (scores != NULL) {
    silhouette_score_per_sample = raft::make_device_vector_view<double, int64_t>(scores, n_rows);
  }

  return cuvs::stats::silhouette_score_batched(
    handle,
    raft::make_device_matrix_view<const double, int64_t>(X, n_rows, n_cols),
    raft::make_device_vector_view<const int, int64_t>(y, n_rows),
    silhouette_score_per_sample,
    n_labels,
    chunk,
    static_cast<cuvs::distance::DistanceType>(metric));
}

}  // namespace Batched
}  // namespace Metrics
}  // namespace ML
