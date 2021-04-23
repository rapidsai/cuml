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

// template <typename value_idx, typename value_t>
void hdbscan(const raft::handle_t &handle, float *X, std::size_t m,
             std::size_t n, raft::distance::DistanceType metric, int k,
             int min_pts, int min_cluster_size,
             hdbscan_output<int64_t, float> *out) {
  HDBSCAN::_fit<int64_t, float>(handle, X, m, n, metric, k, min_pts,
                                min_cluster_size, out);
}

// void hdbscan(const raft::handle_t &handle, const float *X, int m, int n,
//              raft::distance::DistanceType metric, int k, int min_pts,
//              hdbscan_output<int, float> *out);

};  // end namespace ML