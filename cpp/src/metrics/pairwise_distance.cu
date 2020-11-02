
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
#include <metrics/pairwiseDistance.cuh>

namespace ML {

namespace Metrics {
void pairwiseDistance(const raft::handle_t &handle, const double *x,
                      const double *y, double *dist, int m, int n, int k,
                      ML::Distance::DistanceType metric, bool isRowMajor) {
  MLCommon::Metrics::pairwiseDistance(x, y, dist, m, n, k, metric,
                                      handle.get_device_allocator(),
                                      handle.get_stream(), isRowMajor);
}

void pairwiseDistance(const raft::handle_t &handle, const float *x,
                      const float *y, float *dist, int m, int n, int k,
                      ML::Distance::DistanceType metric, bool isRowMajor) {
  MLCommon::Metrics::pairwiseDistance(x, y, dist, m, n, k, metric,
                                      handle.get_device_allocator(),
                                      handle.get_stream(), isRowMajor);
}

}  // namespace Metrics
}  // namespace ML
