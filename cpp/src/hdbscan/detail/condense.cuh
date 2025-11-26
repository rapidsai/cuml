/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels/condense.cuh"

#include <cuml/cluster/hdbscan.hpp>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

#include <algorithm>
#include <cstdint>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Condense {

template <typename value_idx, typename value_t>
value_t get_lambda_host(value_idx node, value_idx num_points, const value_t* deltas)
{
  value_t delta = deltas[node - num_points];
  return delta > 0.0 ? 1.0 / delta : std::numeric_limits<value_t>::max();
}

/* Heuristic dispatching to bottom-up implementation. A high persistent_ratio means there are more
chances to stop early as we climb up the tree, making it more efficient for bottom-up approach
Note: These thresholds are based on empirical observations from running the algorithm on
datasets ranging from 30K to 5M points, with varying distributions and thread counts. For more
details, please refer to the tables in this PR: https://github.com/rapidsai/cuml/pull/7459
*/
bool dispatch_to_bottom_up(int num_persistent, int n_leaves)
{
  int n_nodes            = 2 * n_leaves - 1;
  int internal_nodes     = n_nodes - n_leaves;
  float persistent_ratio = static_cast<float>(num_persistent) / static_cast<float>(internal_nodes);
  int num_omp_threads    = omp_get_max_threads();
  if (persistent_ratio >= 0.001) {
    return true;
  } else if (persistent_ratio >= 0.0001 && num_omp_threads >= 16) {
    return true;
  } else if (num_omp_threads >= 64) {
    return true;
  } else {
    return false;
  }
}

/**
 * Bottom-up approach using OpenMP on CPU
 */
template <typename value_idx, typename value_t>
void _build_condensed_hiearchy_bottom_up(const raft::handle_t& handle,
                                         const value_idx* children,
                                         const value_t* delta,
                                         const value_idx* sizes,
                                         const rmm::device_uvector<uint8_t>& is_persistent,
                                         int n_leaves,
                                         int min_cluster_size,
                                         rmm::device_uvector<value_idx>& out_parent,
                                         rmm::device_uvector<value_idx>& out_child,
                                         rmm::device_uvector<value_t>& out_lambda,
                                         rmm::device_uvector<value_idx>& out_size)
{
  cudaStream_t stream = handle.get_stream();
  int total_internal  = n_leaves - 1;

  value_idx root    = 2 * (n_leaves - 1);
  value_idx n_nodes = 2 * n_leaves - 1;

  std::vector<value_idx> h_children(2 * (n_leaves - 1));
  std::vector<value_t> h_delta(n_leaves - 1);
  std::vector<value_idx> h_sizes(n_leaves - 1);
  std::vector<uint8_t> h_is_persistent(total_internal, 0);

  raft::copy(h_children.data(), children, 2 * (n_leaves - 1), stream);
  raft::copy(h_delta.data(), delta, n_leaves - 1, stream);
  raft::copy(h_sizes.data(), sizes, n_leaves - 1, stream);
  raft::copy(h_is_persistent.data(), is_persistent.data(), total_internal, stream);
  handle.sync_stream(stream);

  // Allocate host output arrays
  std::vector<value_idx> h_out_parent((root + 1) * 2, -1);
  std::vector<value_idx> h_out_child((root + 1) * 2, -1);
  std::vector<value_t> h_out_lambda((root + 1) * 2, -1);
  std::vector<value_idx> h_out_sizes((root + 1) * 2, -1);

  {
    std::vector<int> parent(n_nodes, -1);

    /*
     * Get parent information for easy pointer chasing.
     */
#pragma omp parallel for
    for (int node = n_leaves; node < n_nodes; node++) {
      int left      = h_children[(node - n_leaves) * 2];
      int right     = h_children[(node - n_leaves) * 2 + 1];
      parent[left]  = node;
      parent[right] = node;
    }

    std::vector<value_idx> relabel(n_nodes, 0);

    /*
     * Build the parent–child relationships between internal and leaf nodes,
     * and assign each leaf a representative ancestor and a lambda value.
     *
     *   - Every leaf node should be connected to an internal node
     *     that lies directly below the nearest persistent node in the hierarchy.
     *   - The `relabel` array stores this mapping: for each node, it records the
     *     ancestor that serves as its cluster representative.
     *  - The first internal ancestor whose size >= min_cluster_size provides
     *     the lambda value for the leaf.
     */
#pragma omp parallel for
    for (int node = 0; node < n_nodes; node++) {
      int current         = node;
      int ancestor        = parent[current];
      int rel             = -1;
      float assign_lambda = -1.0f;

      // climb until we reach root or a persistent parent
      while (ancestor != -1) {
        if (node < n_leaves && assign_lambda == -1 &&
            h_sizes[ancestor - n_leaves] >= min_cluster_size) {
          // if leaf node, keep track of delta value of the first internal node that has >= min
          // clusters
          assign_lambda = get_lambda_host(ancestor, n_leaves, h_delta.data());
        }
        bool ancestor_persistent = (ancestor >= n_leaves && h_is_persistent[ancestor - n_leaves]);

        if (ancestor_persistent) {
          rel = current;
          break;
        }

        current  = ancestor;
        ancestor = parent[current];
      }
      // ancestor stays -1 for the root
      relabel[node] = ancestor == -1 ? root : rel;

      if (node < n_leaves) {
        h_out_parent[node * 2] = relabel[node];
        h_out_child[node * 2]  = node;
        h_out_lambda[node * 2] = assign_lambda;
        h_out_sizes[node * 2]  = 1;
      }
    }

    /*
     * Add edges for internal (persistent) nodes to their children using the relabel array.
     */
#pragma omp parallel for
    for (int node = n_leaves; node < n_nodes; node++) {
      if (h_is_persistent[node - n_leaves]) {
        value_idx left_child  = h_children[(node - n_leaves) * 2];
        value_idx right_child = h_children[((node - n_leaves) * 2) + 1];
        int left_count        = left_child >= n_leaves ? h_sizes[left_child - n_leaves] : 1;
        int right_count       = right_child >= n_leaves ? h_sizes[right_child - n_leaves] : 1;
        value_t lambda_value  = get_lambda_host(node, n_leaves, h_delta.data());

        h_out_parent[node * 2] = relabel[node];
        h_out_child[node * 2]  = left_child;
        h_out_lambda[node * 2] = lambda_value;
        h_out_sizes[node * 2]  = left_count;

        h_out_parent[node * 2 + 1] = relabel[node];
        h_out_child[node * 2 + 1]  = right_child;
        h_out_lambda[node * 2 + 1] = lambda_value;
        h_out_sizes[node * 2 + 1]  = right_count;
      }
    }
  }

  // Copy results back to device
  raft::copy(out_parent.data(), h_out_parent.data(), (root + 1) * 2, stream);
  raft::copy(out_child.data(), h_out_child.data(), (root + 1) * 2, stream);
  raft::copy(out_lambda.data(), h_out_lambda.data(), (root + 1) * 2, stream);
  raft::copy(out_size.data(), h_out_sizes.data(), (root + 1) * 2, stream);
}

/**
 * Top-down BFS approach on GPU
 */
template <typename value_idx, typename value_t, int tpb = 256>
void _build_condensed_hierarchy_top_down(const raft::handle_t& handle,
                                         const value_idx* children,
                                         const value_t* delta,
                                         const value_idx* sizes,
                                         int n_leaves,
                                         int min_cluster_size,
                                         rmm::device_uvector<value_idx>& out_parent,
                                         rmm::device_uvector<value_idx>& out_child,
                                         rmm::device_uvector<value_t>& out_lambda,
                                         rmm::device_uvector<value_idx>& out_size)
{
  cudaStream_t stream = handle.get_stream();
  auto exec_policy    = handle.get_thrust_policy();

  // Root is the last edge in the dendrogram
  value_idx root = 2 * (n_leaves - 1);

  rmm::device_uvector<bool> frontier(root + 1, stream);
  rmm::device_uvector<bool> next_frontier(root + 1, stream);

  thrust::fill(exec_policy, frontier.begin(), frontier.end(), false);
  thrust::fill(exec_policy, next_frontier.begin(), next_frontier.end(), false);

  // Array to propagate the lambda of subtrees actively being collapsed
  // through multiple bfs iterations.
  rmm::device_uvector<value_t> ignore(root + 1, stream);

  // Propagate labels from root
  rmm::device_uvector<value_idx> relabel(root + 1, stream);
  thrust::fill(exec_policy, relabel.begin(), relabel.end(), -1);

  raft::update_device(relabel.data() + root, &root, 1, stream);

  // Flip frontier for root
  constexpr bool start = true;
  raft::update_device(frontier.data() + root, &start, 1, stream);

  thrust::fill(exec_policy, out_parent.begin(), out_parent.end(), -1);
  thrust::fill(exec_policy, out_child.begin(), out_child.end(), -1);
  thrust::fill(exec_policy, out_lambda.begin(), out_lambda.end(), -1);
  thrust::fill(exec_policy, out_size.begin(), out_size.end(), -1);
  thrust::fill(exec_policy, ignore.begin(), ignore.end(), -1);

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(root + 1, static_cast<value_idx>(tpb));

  value_idx n_elements_to_traverse =
    thrust::reduce(exec_policy, frontier.data(), frontier.data() + root + 1, 0);

  while (n_elements_to_traverse > 0) {
    // TODO: Investigate whether it would be worth performing a gather/argmatch in order
    // to schedule only the number of threads needed. (it might not be worth it)
    condense_hierarchy_kernel<<<grid, tpb, 0, stream>>>(frontier.data(),
                                                        next_frontier.data(),
                                                        ignore.data(),
                                                        relabel.data(),
                                                        children,
                                                        delta,
                                                        sizes,
                                                        n_leaves,
                                                        min_cluster_size,
                                                        out_parent.data(),
                                                        out_child.data(),
                                                        out_lambda.data(),
                                                        out_size.data());

    thrust::copy(exec_policy, next_frontier.begin(), next_frontier.end(), frontier.begin());
    thrust::fill(exec_policy, next_frontier.begin(), next_frontier.end(), false);

    n_elements_to_traverse = thrust::reduce(
      exec_policy, frontier.data(), frontier.data() + root + 1, static_cast<value_idx>(0));
  }
}

/**
 * Condenses a binary single-linkage tree dendrogram in the Scipy hierarchy
 * format by collapsing subtrees that fall below a minimum cluster size.
 *
 * This function dispatches to either a CPU (bottom-up) or GPU (top-down BFS)
 * implementation based on a heuristic that considers the persistent node ratio
 * and available OpenMP threads.
 *
 * For increased parallelism, the output array sizes are held fixed but
 * the result will be sparse (e.g. zeros in place of parents who have been
 * removed / collapsed). This function accepts an empty instance of
 * `CondensedHierarchy` and invokes the `condense()` function on it to
 * convert the sparse output arrays into their dense form.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param handle
 * @param[in] children parents/children from single-linkage dendrogram
 * @param[in] delta distances from single-linkage dendrogram
 * @param[in] sizes sizes from single-linkage dendrogram
 * @param[in] min_cluster_size any subtrees less than this size will be
 *                             collapsed.
 * @param[in] n_leaves number of actual data samples in the dendrogram
 * @param[out] condensed_tree output dendrogram. will likely no longer be
 *                            a binary tree.
 */
template <typename value_idx, typename value_t, int tpb = 256>
void build_condensed_hierarchy(const raft::handle_t& handle,
                               const value_idx* children,
                               const value_t* delta,
                               const value_idx* sizes,
                               int min_cluster_size,
                               int n_leaves,
                               Common::CondensedHierarchy<value_idx, value_t>& condensed_tree)
{
  cudaStream_t stream = handle.get_stream();
  auto exec_policy    = handle.get_thrust_policy();

  // Root is the last edge in the dendrogram
  value_idx root = 2 * (n_leaves - 1);

  auto d_ptr           = thrust::device_pointer_cast(children);
  value_idx n_vertices = *(thrust::max_element(exec_policy, d_ptr, d_ptr + root)) + 1;

  // Prevent potential infinite loop from labeling disconnected
  // connectivities graph.
  RAFT_EXPECTS(n_vertices == root,
               "Multiple components found in MST or MST is invalid. "
               "Cannot find single-linkage solution. Found %d vertices "
               "total.",
               static_cast<int>(n_vertices));

  int total_internal = n_leaves - 1;

  // Identify persistent nodes (those with size >= min_cluster_size)
  rmm::device_uvector<uint8_t> is_persistent(total_internal, stream);
  thrust::fill(exec_policy, is_persistent.begin(), is_persistent.end(), 0);

  size_t grid = raft::ceildiv(total_internal, static_cast<int>(tpb));
  get_persistent_nodes_kernel<<<grid, tpb, 0, stream>>>(
    children, delta, sizes, is_persistent.data(), min_cluster_size, n_leaves);

  thrust::device_ptr<uint8_t> is_persistent_ptr(is_persistent.data());
  int num_persistent =
    thrust::count(exec_policy, is_persistent_ptr, is_persistent_ptr + total_internal, 1);

  // Allocate output arrays
  rmm::device_uvector<value_idx> out_parent((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_child((root + 1) * 2, stream);
  rmm::device_uvector<value_t> out_lambda((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_size((root + 1) * 2, stream);

  // Dispatch to CPU or GPU implementation based on heuristic
  if (dispatch_to_bottom_up(num_persistent, n_leaves)) {
    _build_condensed_hiearchy_bottom_up(handle,
                                        children,
                                        delta,
                                        sizes,
                                        is_persistent,
                                        n_leaves,
                                        min_cluster_size,
                                        out_parent,
                                        out_child,
                                        out_lambda,
                                        out_size);
  } else {
    _build_condensed_hierarchy_top_down<value_idx, value_t, tpb>(handle,
                                                                 children,
                                                                 delta,
                                                                 sizes,
                                                                 n_leaves,
                                                                 min_cluster_size,
                                                                 out_parent,
                                                                 out_child,
                                                                 out_lambda,
                                                                 out_size);
  }

  handle.sync_stream(stream);

  condensed_tree.condense(out_parent.data(), out_child.data(), out_lambda.data(), out_size.data());
}

};  // end namespace Condense
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
