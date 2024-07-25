/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
namespace Select {

/**
 * For each non-0 value in frontier, deselects children clusters
 * and adds children to frontier.
 * @tparam value_idx
 * @param[in] indptr CSR indptr of array (size n_clusters+1)
 * @param[in] children array of children indices (size n_clusters)
 * @param[inout] frontier frontier array storing which nodes need
 *               processing in each kernel invocation (size n_clusters)
 * @param[inout] is_cluster array of cluster selections / deselections (size n_clusters)
 * @param[in] n_clusters number of clusters
 */
template <typename value_idx>
CUML_KERNEL void propagate_cluster_negation_kernel(const value_idx* indptr,
                                                   const value_idx* children,
                                                   int* frontier,
                                                   int* next_frontier,
                                                   int* is_cluster,
                                                   int n_clusters)
{
  int cluster = blockDim.x * blockIdx.x + threadIdx.x;

  if (cluster < n_clusters && frontier[cluster]) {
    frontier[cluster] = false;

    value_idx children_start = indptr[cluster];
    value_idx children_stop  = indptr[cluster + 1];
    for (int i = children_start; i < children_stop; i++) {
      value_idx child      = children[i];
      next_frontier[child] = true;
      is_cluster[child]    = false;
    }
  }
}

template <typename value_idx, typename value_t, int tpb = 256>
CUML_KERNEL void cluster_epsilon_search_kernel(const int* selected_clusters,
                                               const int n_selected_clusters,
                                               const value_idx* parents,
                                               const value_idx* children,
                                               const value_t* lambdas,
                                               const value_idx cluster_tree_edges,
                                               int* is_cluster,
                                               int* frontier,
                                               const int n_clusters,
                                               const value_t cluster_selection_epsilon,
                                               const bool allow_single_cluster)
{
  auto selected_cluster_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (selected_cluster_idx >= n_selected_clusters) { return; }

  // don't need to process root as a cluster
  // offsetting for root by subtracting 1 from the cluster
  // this is because root isn't involved in epsilon search directly
  // further, it allows the remaining clusters to be 0-index
  // and directly access from the children/lambda arrays
  // since parents/lambdas are sorted by children
  // the relation is: child = child_idx + 1
  auto child_idx = selected_clusters[selected_cluster_idx] - 1;
  if (child_idx == -1) { return; }

  auto eps = 1 / lambdas[child_idx];

  if (eps < cluster_selection_epsilon) {
    constexpr auto root = 0;

    value_idx parent;
    value_t parent_eps;

    do {
      parent = parents[child_idx];

      if (parent == root) {
        if (!allow_single_cluster) {
          // setting parent to actual value of child
          // by offsetting
          parent = child_idx + 1;
        }
        break;
      }

      // again, offsetting for root
      child_idx = parent - 1;
      // lambda is picked for where the parent
      // resides according to where it is a child
      parent_eps = 1 / lambdas[child_idx];
    } while (parent_eps <= cluster_selection_epsilon);

    frontier[parent]   = true;
    is_cluster[parent] = true;
  } else {
    // offset 1 ahead for root
    frontier[child_idx + 1] = true;
  }
}

};  // namespace Select
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
