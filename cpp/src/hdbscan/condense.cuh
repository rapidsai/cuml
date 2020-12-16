/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <cuml/cuml_api.h>
#include <raft/cudart_utils.h>
#include <common/cumlHandle.hpp>

#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/mst/mst.cuh>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <distance/distance.cuh>

#include <cuml/neighbors/knn.hpp>

namespace ML {
namespace HDBSCAN {
namespace Condense {

template <typename value_idx, typename value_t>
void bfs_from_hierarchy() {}

template <typename value_idx, typename value_t>
void condense(value_idx *tree_src, value_idx *tree_dst, value_t *tree_delta,
              value_idx *tree_size, value_idx m) {
  value_idx root = 2 * m;

  value_idx n_points = root / 2 + 1;
  value_idx next_label = n_points + 1;
}

};  // end namespace Condense
};  // end namespace HDBSCAN
};  // end namespace ML