
/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <raft/stats/adjusted_rand_index.cuh>

namespace ML {

namespace Metrics {
double adjusted_rand_index(const raft::handle_t& handle,
                           const int64_t* y,
                           const int64_t* y_hat,
                           const int64_t n)
{
  return raft::stats::adjusted_rand_index<int64_t, unsigned long long>(
    y, y_hat, n, handle.get_stream());
}

double adjusted_rand_index(const raft::handle_t& handle,
                           const int* y,
                           const int* y_hat,
                           const int n)
{
  return raft::stats::adjusted_rand_index<int, unsigned long long>(
    y, y_hat, n, handle.get_stream());
}
}  // namespace Metrics
}  // namespace ML
