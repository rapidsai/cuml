/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/util/fast_int_div.cuh>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Predict {

template <typename value_idx, typename value_t, int tpb = 256>
CUML_KERNEL void merge_height_kernel(value_t* heights,
                                     value_t* lambdas,
                                     value_idx* index_into_children,
                                     value_idx* parents,
                                     size_t m,
                                     value_idx n_selected_clusters,
                                     raft::util::FastIntDiv n,
                                     value_idx* selected_clusters)
{
  value_idx idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < value_idx(m * n_selected_clusters)) {
    value_idx row           = idx / n_selected_clusters;
    value_idx col           = idx % n_selected_clusters;
    value_idx right_cluster = selected_clusters[col];
    value_idx left_cluster  = parents[index_into_children[row]];
    bool took_right_parent  = false;
    bool took_left_parent   = false;
    value_idx last_cluster;

    while (left_cluster != right_cluster) {
      if (left_cluster > right_cluster) {
        took_left_parent = true;
        last_cluster     = left_cluster;
        left_cluster     = parents[index_into_children[left_cluster]];
      } else {
        took_right_parent = true;
        last_cluster      = right_cluster;
        right_cluster     = parents[index_into_children[right_cluster]];
      }
    }

    if (took_left_parent && took_right_parent) {
      heights[idx] = lambdas[index_into_children[last_cluster]];
    }

    else {
      heights[idx] = lambdas[index_into_children[row]];
    }
  }
}

template <typename value_idx, typename value_t, int tpb = 256>
CUML_KERNEL void merge_height_kernel(value_t* heights,
                                     value_t* lambdas,
                                     value_t* prediction_lambdas,
                                     value_idx* min_mr_indices,
                                     value_idx* index_into_children,
                                     value_idx* parents,
                                     size_t n_prediction_points,
                                     value_idx n_selected_clusters,
                                     raft::util::FastIntDiv n,
                                     value_idx* selected_clusters)
{
  value_idx idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < value_idx(n_prediction_points * n_selected_clusters)) {
    value_idx row           = idx / n;
    value_idx col           = idx % n;
    value_idx right_cluster = selected_clusters[col];
    value_idx left_cluster  = parents[index_into_children[min_mr_indices[row]]];
    bool took_right_parent  = false;
    bool took_left_parent   = false;
    value_idx last_cluster;

    while (left_cluster != right_cluster) {
      if (left_cluster > right_cluster) {
        took_left_parent = true;
        last_cluster     = left_cluster;
        left_cluster     = parents[index_into_children[left_cluster]];
      } else {
        took_right_parent = true;
        last_cluster      = right_cluster;
        right_cluster     = parents[index_into_children[right_cluster]];
      }
    }

    if (took_left_parent && took_right_parent) {
      heights[idx] = lambdas[index_into_children[last_cluster]];
    }

    else {
      heights[idx] = prediction_lambdas[row];
    }
  }
}

};  // namespace Predict
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
