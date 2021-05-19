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
#include <rmm/exec_policy.hpp>

#include <raft/sparse/op/sort.h>
#include <raft/sparse/convert/csr.cuh>

#include <cuml/cluster/hdbscan.hpp>

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
        //        printf("persisting node=%d, left_child=%d, left_size=%d, right_child=%d, right_size=%d\n",
        //               node, left_child, left_count, right_child, right_count);

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
  auto exec_policy = rmm::exec_policy(stream);

  // Root is the last edge in the dendrogram
  int root = 2 * (n_leaves - 1);

  auto d_ptr = thrust::device_pointer_cast(children);
  value_idx n_vertices =
    *(thrust::max_element(exec_policy, d_ptr, d_ptr + root)) + 1;

  // Prevent potential infinite loop from labeling disconnected
  // connectivities graph.
  RAFT_EXPECTS(n_vertices == (n_leaves - 1) * 2,
               "Multiple components found in MST or MST is invalid. "
               "Cannot find single-linkage solution.");

  rmm::device_uvector<bool> frontier(root + 1, stream);

  thrust::fill(exec_policy, frontier.data(), frontier.data() + frontier.size(),
               false);

  rmm::device_uvector<value_t> ignore(root + 1, stream);

  // Propagate labels from root
  rmm::device_uvector<value_idx> relabel(root + 1, handle.get_stream());
  thrust::fill(exec_policy, relabel.data(), relabel.data() + relabel.size(),
               -1);

  raft::update_device(relabel.data() + root, &root, 1, handle.get_stream());

  // Flip frontier for root
  bool start = true;
  raft::update_device(frontier.data() + root, &start, 1, handle.get_stream());

  rmm::device_uvector<value_idx> out_parent((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_child((root + 1) * 2, stream);
  rmm::device_uvector<value_t> out_lambda((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_size((root + 1) * 2, stream);

  thrust::fill(exec_policy, out_parent.data(),
               out_parent.data() + out_parent.size(), -1);
  thrust::fill(exec_policy, out_child.data(),
               out_child.data() + out_child.size(), -1);
  thrust::fill(exec_policy, out_lambda.data(),
               out_lambda.data() + out_lambda.size(), -1);
  thrust::fill(exec_policy, out_size.data(), out_size.data() + out_size.size(),
               -1);
  thrust::fill(exec_policy, ignore.data(), ignore.data() + ignore.size(), -1);

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(root + 1, (int)tpb);

  value_idx n_elements_to_traverse =
    thrust::reduce(exec_policy, frontier.data(), frontier.data() + root + 1, 0);

  while (n_elements_to_traverse > 0) {
    // TODO: Investigate whether it would be worth performing a gather/argmatch in order
    // to schedule only the number of threads needed. (it might not be worth it)
    condense_hierarchy_kernel<<<grid, tpb, 0, handle.get_stream()>>>(
      frontier.data(), ignore.data(), relabel.data(), children, delta, sizes,
      n_leaves, min_cluster_size, out_parent.data(), out_child.data(),
      out_lambda.data(), out_size.data());

    n_elements_to_traverse = thrust::reduce(exec_policy, frontier.data(),
                                            frontier.data() + root + 1, 0);

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  condensed_tree.condense(out_parent.data(), out_child.data(),
                          out_lambda.data(), out_size.data());
}

};  // end namespace Condense
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML