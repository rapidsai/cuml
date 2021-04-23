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
namespace Tree {
namespace detail {

template <typename value_idx, typename value_t>
__device__ value_t get_lambda(value_idx node, value_idx num_points,
                                value_t *deltas) {
    value_t delta = deltas[node - num_points];
    if (delta > 0.0) return 1.0 / delta;
    return std::numeric_limits<value_t>::max();
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
    bool *frontier, value_idx *ignore, value_idx *relabel,
    const value_idx *src, const value_idx *dst, const value_t *deltas,
    const value_idx *sizes, int n_leaves,
    int num_points, int min_cluster_size,
    value_idx *out_parent, value_idx *out_child,
    value_t *out_lambda, value_idx *out_count) {

    int node = blockDim.x * blockIdx.x + threadIdx.x;

    // If node is in frontier, flip frontier for children
    if(node > n_leaves * 2 || !frontier[node])
    return;

    frontier[node] = false;

    // TODO: Check bounds
    value_idx left_child = src[(node - num_points) * 2];
    value_idx right_child = dst[((node - num_points) * 2)];

    frontier[left_child] = true;
    frontier[right_child] = true;

    bool ignore_val = ignore[node];
    bool should_ignore = ignore_val > -1;

    // If the current node is being ignored (e.g. > -1) then propagate the ignore
    // to children, if any
    ignore[left_child] = (should_ignore * ignore_val) + (!should_ignore * -1);
    ignore[right_child] = (should_ignore * ignore_val) + (!should_ignore * -1);

    if (node < num_points) {
    out_parent[node] = relabel[should_ignore];
    out_child[node] = node;
    out_lambda[node] = get_lambda(should_ignore, num_points, deltas);
    out_count[node] = 1;
    }

    // If node is not ignored and is not a leaf, condense its children
    // if necessary
    else if (!should_ignore and node >= num_points) {
    value_idx left_child = src[(node - num_points) * 2];
    value_idx right_child = dst[((node - num_points) * 2)];

    value_t lambda_value = get_lambda(node, num_points, deltas);

    int left_count =
        left_child >= num_points ? sizes[left_child - num_points] : 1;
    int right_count =
        right_child >= num_points ? sizes[right_child - num_points] : 1;

    // If both children are large enough, they should be relabeled and
    // included directly in the output hierarchy.
    if (left_count >= min_cluster_size && right_count >= min_cluster_size) {
        relabel[left_child] = node;
        out_parent[node] = relabel[node];
        out_child[node] = node;
        out_lambda[node] = lambda_value;
        out_count[node] = left_count;

        relabel[right_child] = node;
        out_parent[node] = relabel[node];
        out_child[node] = node;
        out_lambda[node] = lambda_value;
        out_count[node] = left_count;
    }

    // Consume left or right child as necessary
    bool left_child_too_small = left_count < min_cluster_size;
    bool right_child_too_small = right_count < min_cluster_size;
    ignore[left_child] =
        (left_child_too_small * node) + (!left_child_too_small * -1);
    ignore[right_child] =
        (right_child_too_small * node) + (!right_child_too_small * -1);

    // If only left or right child is too small, consume it and relabel the other
    // (to it can be its own cluster)
    bool only_left_child_too_small =
        left_child_too_small && !right_child_too_small;
    bool only_right_child_too_small =
        !left_child_too_small && right_child_too_small;

    relabel[right_child] = (only_left_child_too_small * relabel[node]) +
                            (!only_left_child_too_small * -1);
    relabel[left_child] = (only_right_child_too_small * relabel[node]) +
                            (!only_right_child_too_small * -1);
    }
}

template<typename value_idx, typename value_t>
__global__ void propagate_cluster_negation(const value_idx *indptr,
                                           const value_idx *children,
                                           bool *frontier,
                                           bool *is_cluster,
                                           int n_clusters) {

  int cluster = blockDim.x * blockIdx.x + threadIdx.x;

  if(cluster < n_clusters && frontier[cluster]) {
    frontier[cluster] = false;

    value_idx children_start = indptr[cluster];
    value_idx children_stop = indptr[cluster];
    for(int i = 0; i < children_stop - children_start; i++) {

      value_idx child = children[i];
      frontier[child] = true;
      is_cluster[child] = false;
    }
  }
}
    

};  // end namespace detail
};  // end namespace Tree
};  // end namespace HDBSCAN
};  // end namespace ML