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

#include <raft/cudart_utils.h>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace ML {
namespace HDBSCAN {
namespace Tree {

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
  value_idx *hierarchy, value_t *deltas, value_idx *sizes, int n_leaves,
  int num_points, int min_cluster_size) {
  int node = blockDim.x * blockIdx.x + threadIdx.x;

  // If node is in frontier, flip frontier for children
  if (node <= n_leaves * 2 && frontier[node]) {
    frontier[node] = false;

    // TODO: Check bounds
    value_idx left_child = hierarchy[(node - num_points) * 2];
    value_idx right_child = hierarchy[((node - num_points) * 2) + 1];

    frontier[left_child] = true;
    frontier[right_child] = true;

    bool ignore_val = ignore[node];
    bool should_ignore = ignore_val > -1;

    // If the current node is being ignored (e.g. > -1) then propagate the ignore
    // to children, if any
    ignore[left_child] = (should_ignore * ignore_val) + (!should_ignore * -1);
    ignore[right_child] = (should_ignore * ignore_val) + (!should_ignore * -1);

    if (node < num_points) {
      // TODO: append relabel[should_ignore], node, get_lambda(should_ignore), 1

    }

    // If node is not ignored and is not a leaf, condense its children
    // if necessary
    else if (!should_ignore and node >= num_points) {
      value_idx left_child = hierarchy[(node - num_points) * 2];
      value_idx right_child = hierarchy[((node - num_points) * 2) + 1];

      value_t lambda_value = get_lambda(node, num_points, deltas);

      // TODO: Convert to boolean arithmetic
      int left_count =
        left_child >= num_points ? sizes[left_child - num_points] : 1;
      int right_count =
        right_child >= num_points ? sizes[right_child - num_points] : 1;

      // If both children are large enough, they should be relabeled and
      // included directly in the output hierarchy.
      if (left_count >= min_cluster_size && right_count >= min_cluster_size) {
        relabel[left_child] = node;
        // TODO: Output new hierarchy entry for: relabel[node], relabel[left], lambda_value, left_count

        relabel[right_child] = node;
        // TODO Output new hierarchy entry for: relabel[node], relabel[right], lambda_value, right_count
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
}

template <typename value_idx, typename value_t, int tpb = 256>
void condense_hierarchy(raft::handle_t &handle, value_idx *children,
                        value_t *delta, value_idx *sizes, int min_pts,
                        int n_leaves) {
  rmm::device_uvector<bool> frontier(n_leaves * 2, handle.get_stream());
  rmm::device_uvector<bool> ignore(n_leaves * 2, handle.get_stream());

  int root = 2 * n_leaves;
  int num_points = floor(root / 2.0) + 1;

  rmm::device_uvector<value_idx> relabel(root + 1, handle.get_stream());

  // TODO: Set this properly on device
  relabel[root] = root;

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(n_leaves * 2, (size_t)tpb);

  value_idx n_elements_to_traverse =
    thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                   frontier.data() + (n_leaves * 2), 0);

  while (n_elements_to_traverse > 0) {
    condense_hierarchy_kernel<<<grid, tpb, 0, handle.get_stream()>>>(
      frontier.data(), ignore.data(), next_label.data(), relabel.data(),
      children, delta, sizes, n_leaves, num_points, min_cluster_size);

    n_elements_to_traverse =
      thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                     frontier.data() + (n_leaves * 2), 0);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  }

  // TODO: Normalize labels so they are drawn from a monotonically increasing set.
}

template<typename value_idx, typename value_t>
void compute_stabilities(value_idx *condensed_hierarchy, value_idx *lambdas, value_idx *sizes,
                         int n_leaves) {

  // TODO: Reverse topological sort (e.g. sort hierarchy, lambdas, and sizes by lambda)

  // TODO: Segmented reduction on min_lambda within each cluster

  // TODO: Embarassingly parallel construction of output
}
};  // end namespace Tree
};  // end namespace HDBSCAN
};  // end namespace ML