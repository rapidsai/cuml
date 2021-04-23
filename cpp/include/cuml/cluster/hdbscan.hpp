/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#pragma once

#include <raft/linalg/distance_type.h>

#include <cuml/cuml.hpp>

#include <cstddef>

namespace ML {

template <typename value_idx, typename value_t>
struct hdbscan_output {
  int n_clusters;
  value_idx *labels;
  value_t *probabilities;
};

// template <typename value_idx = int64_t, typename value_t = float>
void hdbscan(const raft::handle_t &handle, float *X, std::size_t m,
             std::size_t n, raft::distance::DistanceType metric, int k,
             int min_pts, int min_cluster_size,
             hdbscan_output<int64_t, float> *out);
};  // end namespace ML