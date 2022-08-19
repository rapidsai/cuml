/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <metrics/davies_bouldin_score.cuh>
#include <raft/distance/distance_type.hpp>

namespace ML {

namespace Metrics {
double davies_bouldin_score(const raft::handle_t& handle,
                            double* y,
                            int nRows,
                            int nCols,
                            int* labels,
                            int nLabels,
                            raft::distance::DistanceType metric)
{
  return MLCommon::Metrics::davies_bouldin_score<double, int>(
    handle, y, nRows, nCols, labels, nLabels, handle.get_stream(), metric);
}


}  // namespace Metrics
}  // namespace ML
