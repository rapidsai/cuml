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

#pragma once

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Condense {

template <typename value_idx, typename value_t>
__device__ inline value_t get_lambda(value_idx node, value_idx num_points,
                                     const value_t *deltas) {
  value_t delta = deltas[node - num_points];
  bool nonzero_delta = delta > 0.0;
  return ((nonzero_delta * 1.0) / (nonzero_delta * delta)) +
         (!nonzero_delta * std::numeric_limits<value_t>::max());
}

/**
    *
    * @tparam value_idx
    * @tparam value_t
    * @param frontier
    * @param ignore Should be initialized to -1
    * @param next_label
    * @param relabel
    * @param hierarchy
    * @param deltas
    * @param sizes
    * @param n_leaves
    * @param num_points
    * @param min_cluster_size
    */
template <typename value_idx, typename value_t>
__global__ void condense_hierarchy_kernel(
  bool *frontier, value_t *ignore, value_idx *relabel,
  const value_idx *children, const value_t *deltas, const value_idx *sizes,
  int n_leaves, int min_cluster_size, value_idx *out_parent,
  value_idx *out_child, value_t *out_lambda, value_idx *out_count) {
  int node = blockDim.x * blockIdx.x + threadIdx.x;

  if (node >= n_leaves * 2 - 1 || !frontier[node]) return;

  frontier[node] = false;

  value_t subtree_lambda = ignore[node];

  bool should_ignore = subtree_lambda > -1;

  // TODO: Investigate whether this would be better done w/ an additional
  //  kernel launch

  // If node is a leaf, add it to the condensed hierarchy
  if (node < n_leaves) {
    out_parent[node * 2] = relabel[node];
    out_child[node * 2] = node;
    out_lambda[node * 2] = subtree_lambda;
    out_count[node * 2] = 1;
  }

  // If node is not a leaf, condense its children if necessary
  else {
    value_idx left_child = children[(node - n_leaves) * 2];
    value_idx right_child = children[((node - n_leaves) * 2) + 1];

    // flip frontier for children
    frontier[left_child] = true;
    frontier[right_child] = true;

    // propagate ignore down to children
    ignore[left_child] =
      (should_ignore * subtree_lambda) + (!should_ignore * -1);
    ignore[right_child] =
      (should_ignore * subtree_lambda) + (!should_ignore * -1);

    relabel[left_child] =
      (should_ignore * relabel[node]) + (!should_ignore * relabel[left_child]);
    relabel[right_child] =
      (should_ignore * relabel[node]) + (!should_ignore * relabel[right_child]);

    value_idx node_relabel = relabel[node];

    // TODO: Should be able to remove this nested conditional
    if (!should_ignore) {
      value_t lambda_value = get_lambda(node, n_leaves, deltas);

      int left_count =
        left_child >= n_leaves ? sizes[left_child - n_leaves] : 1;
      int right_count =
        right_child >= n_leaves ? sizes[right_child - n_leaves] : 1;

      // Consume left or right child as necessary
      bool left_child_too_small = left_count < min_cluster_size;
      bool right_child_too_small = right_count < min_cluster_size;

      // Node can "persist" to the cluster tree only if
      // both children >= min_cluster_size
      bool can_persist = !left_child_too_small && !right_child_too_small;

      relabel[left_child] =
        (!can_persist * node_relabel) + (can_persist * left_child);
      relabel[right_child] =
        (!can_persist * node_relabel) + (can_persist * right_child);

      // set ignore for children. This is the node at which the "points underneath fall out"
      ignore[left_child] =
        (left_child_too_small * lambda_value) + (!left_child_too_small * -1);
      ignore[right_child] =
        (right_child_too_small * lambda_value) + (!right_child_too_small * -1);

      // If both children are large enough, they should be relabeled and
      // included directly in the output hierarchy.
      if (can_persist) {
        // TODO: Could probably pull this out if this conditional becomes
        //  a bottleneck
        out_parent[node * 2] = node_relabel;
        out_child[node * 2] = left_child;
        out_lambda[node * 2] = lambda_value;
        out_count[node * 2] = left_count;

        out_parent[node * 2 + 1] = node_relabel;
        out_child[node * 2 + 1] = right_child;
        out_lambda[node * 2 + 1] = lambda_value;
        out_count[node * 2 + 1] = right_count;
      }
    }
  }
}

};  // end namespace Condense
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML