
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

#include <cuml/metrics/metrics.hpp>
#include <metrics/silhouetteScore.cuh>

namespace ML {

namespace Metrics {
double silhouetteScore(const raft::handle_t &handle, double *y, int nRows,
                       int nCols, int *labels, int nLabels, double *silScores,
                       int metric) {
  return MLCommon::Metrics::silhouetteScore<double, int>(
    y, nRows, nCols, labels, nLabels, silScores, handle.get_device_allocator(),
    handle.get_stream(), metric);
}
}  // namespace Metrics
}  // namespace ML
