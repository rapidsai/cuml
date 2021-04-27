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

#include <label/classlabels.cuh>

#include <cub/cub.cuh>

#include <raft/cudart_utils.h>

#include <rmm/device_uvector.hpp>

#include <raft/sparse/op/sort.h>
#include <raft/sparse/convert/csr.cuh>

#include "common.h"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

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
  bool *frontier, value_idx *ignore, value_idx *relabel,
  const value_idx *children, const value_t *deltas, const value_idx *sizes,
  int n_leaves, int min_cluster_size, value_idx *out_parent,
  value_idx *out_child, value_t *out_lambda, value_idx *out_count) {
  int node = blockDim.x * blockIdx.x + threadIdx.x;

  if (node >= n_leaves * 2 - 1 || !frontier[node]) return;

  frontier[node] = false;

  value_idx subtree_parent = ignore[node];

  bool should_ignore = subtree_parent > -1;

  // TODO: Investigate whether this would be better done w/ an additional kernel launch

  // If node is a leaf, add it to the condensed hierarchy
  if (node < n_leaves) {
    printf("Leaf node: %d, parent=%d, ignore_val=%d\n", node, subtree_parent, subtree_parent);
    out_parent[node] = subtree_parent;
    out_child[node] = node;
    out_lambda[node] = get_lambda(subtree_parent, n_leaves, deltas);
    out_count[node] = 1;
  }

  // If node is not a leaf, condense its children if necessary
  else {
    value_idx left_child = children[(node - n_leaves) * 2];
    value_idx right_child = children[((node - n_leaves) * 2) + 1];

    // flip frontier for children
    frontier[left_child] = true;
    frontier[right_child] = true;

    ignore[left_child] = (should_ignore * subtree_parent) + (!should_ignore * -1);
    ignore[right_child] = (should_ignore * subtree_parent) + (!should_ignore * -1);

    // TODO: Should be able to remove this nested conditional
    if (!should_ignore) {
      value_t lambda_value = get_lambda(node, n_leaves, deltas);

      int left_count =
        left_child >= n_leaves ? sizes[left_child - n_leaves] : 1;
      int right_count =
        right_child >= n_leaves ? sizes[right_child - n_leaves] : 1;

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
}

/**
 * Condenses a binary tree dendrogram in the Scipy format
 * by merging labels that fall below a minimum cluster size.
 * This function accepts an empty instance of `CondensedHierarchy`
 * and invokes the `condense()` function on it.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param handle
 * @param[in] children
 * @param[in] delta
 * @param[in] sizes
 * @param[in] min_cluster_size
 * @param[in] n_leaves
 * @param[out] out_parent
 * @param[out] out_child
 * @param[out] out_lambda
 * @param[out] out_size
 */
template <typename value_idx, typename value_t, int tpb = 256>
void build_condensed_hierarchy(
  const raft::handle_t &handle, const value_idx *children, const value_t *delta,
  const value_idx *sizes, int min_cluster_size, int n_leaves,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree) {
  cudaStream_t stream = handle.get_stream();

  // Root is the last edge in the dendrogram
  int root = 2 * (n_leaves - 1);

  rmm::device_uvector<bool> frontier(root + 1, stream);

  thrust::fill(thrust::cuda::par.on(stream), frontier.data(),
               frontier.data() + frontier.size(), false);

  rmm::device_uvector<value_idx> ignore(root + 1, stream);

  // Propagate labels from root
  rmm::device_uvector<value_idx> relabel(root + 1, handle.get_stream());
  thrust::fill(thrust::cuda::par.on(stream), relabel.data(),
               relabel.data() + relabel.size(), -1);


  raft::update_device(relabel.data() + root, &root, 1, handle.get_stream());

  // Flip frontier for root
  bool start = true;
  raft::update_device(frontier.data() + root, &start, 1, handle.get_stream());

  rmm::device_uvector<value_idx> out_parent(root, stream);
  rmm::device_uvector<value_idx> out_child(root, stream);
  rmm::device_uvector<value_t> out_lambda(root, stream);
  rmm::device_uvector<value_idx> out_size(root, stream);

  thrust::fill(thrust::cuda::par.on(stream), out_parent.data(),
               out_parent.data() + root, -1);
  thrust::fill(thrust::cuda::par.on(stream), out_child.data(),
               out_child.data() + root, -1);
  thrust::fill(thrust::cuda::par.on(stream), out_lambda.data(),
               out_lambda.data() + root, -1);
  thrust::fill(thrust::cuda::par.on(stream), out_size.data(),
               out_size.data() + root, -1);
  thrust::fill(thrust::cuda::par.on(stream), ignore.data(),
               ignore.data() + ignore.size(), -1);

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(root + 1, (int)tpb);

  value_idx n_elements_to_traverse =
    thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                   frontier.data() + root + 1, 0);

  while (n_elements_to_traverse > 0) {
    // TODO: Investigate whether it would be worth performing a gather/argmatch in order
    // to schedule only the number of threads needed. (it might not be worth it)
    condense_hierarchy_kernel<<<grid, tpb, 0, handle.get_stream()>>>(
      frontier.data(), ignore.data(), relabel.data(), children, delta, sizes,
      n_leaves, min_cluster_size, out_parent.data(),
      out_child.data(), out_lambda.data(), out_size.data());

    n_elements_to_traverse =
      thrust::reduce(thrust::cuda::par.on(handle.get_stream()), frontier.data(),
                     frontier.data() + root + 1, 0);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    raft::print_device_vector("parent", out_parent.data(), out_parent.size(), std::cout);
    raft::print_device_vector("child", out_child.data(), out_child.size(), std::cout);
    raft::print_device_vector("size", out_size.data(), out_size.size(), std::cout);
    raft::print_device_vector("lambda", out_lambda.data(), out_lambda.size(), std::cout);

    raft::print_device_vector("relabel", relabel.data(), relabel.size(), std::cout);
    raft::print_device_vector("ignore", ignore.data(), ignore.size(), std::cout);
  }

  // TODO: Verify the sequence of condensed cluster labels enables topological sort

  condensed_tree.condense(out_parent.data(), out_child.data(),
                          out_lambda.data(), out_size.data());
}

};  // end namespace Condense
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML