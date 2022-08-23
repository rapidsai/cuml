/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Predict {

template <typename value_idx, typename value_t, int tpb = 256>
__global__ void merge_height_kernel(value_t* heights,
                                    value_t* lambdas,
                                    value_idx* index_into_children,
                                    value_idx* parents,
                                    size_t m,
                                    value_idx n_selected_clusters,
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

template <typename value_idx, typename value_t>
__global__ void prob_in_some_cluster_kernel(value_t* heights,
                                            value_t* height_argmax,
                                            value_t* deaths,
                                            value_idx* index_into_children,
                                            value_idx* selected_clusters,
                                            value_t* lambdas,
                                            value_t* prob_in_some_cluster,
                                            value_idx n_selected_clusters,
                                            value_idx n_leaves,
                                            size_t m)
{
  value_idx idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < (value_idx)m) {
    value_t max_lambda = max(lambdas[index_into_children[idx]],
                             deaths[selected_clusters[(int)height_argmax[idx]] - n_leaves]);
    prob_in_some_cluster[idx] =
      heights[idx * n_selected_clusters + (int)height_argmax[idx]] / max_lambda;
    return;
  }
}

};  // namespace Predict
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
