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
#include <raft/sparse/hierarchy/common.h>

#include <cuml/cuml.hpp>

namespace ML {

template <typename value_idx = int64_t, typename value_t = float>
void hdbscan(const raft::handle_t &handle, value_t *X, size_t m, size_t n,
             raft::distance::DistanceType metric, int k, int min_pts,
             float alpha, hdbscan_output<value_idx, value_t> *out);
};  // end namespace ML