/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuml/cluster/hdbscan.hpp>

#include <hdbscan/runner.h>

namespace ML {

void hdbscan(const raft::handle_t &handle, const float *X, size_t m, size_t n,
             raft::distance::DistanceType metric,
             HDBSCAN::Common::HDBSCANParams &params,
             HDBSCAN::Common::hdbscan_output<int, float> &out) {
  HDBSCAN::_fit_hdbscan(handle, X, m, n, metric, params, out);
}

};  // end namespace ML