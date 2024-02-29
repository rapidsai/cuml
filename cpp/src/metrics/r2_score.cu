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
#include <raft/stats/r2_score.cuh>

namespace ML {

namespace Metrics {

float r2_score_py(const raft::handle_t& handle, float* y, float* y_hat, int n)
{
  return raft::stats::r2_score(y, y_hat, n, handle.get_stream());
}

double r2_score_py(const raft::handle_t& handle, double* y, double* y_hat, int n)
{
  return raft::stats::r2_score(y, y_hat, n, handle.get_stream());
}

}  // namespace Metrics
}  // namespace ML
