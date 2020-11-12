
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
#include <metrics/homogeneity_score.cuh>

namespace ML {

namespace Metrics {

double completeness_score(const raft::handle_t &handle, const int *y,
                          const int *y_hat, const int n,
                          const int lower_class_range,
                          const int upper_class_range) {
  return MLCommon::Metrics::homogeneity_score(
    y_hat, y, n, lower_class_range, upper_class_range,
    handle.get_device_allocator(), handle.get_stream());
}

}  // namespace Metrics
}  // namespace ML
