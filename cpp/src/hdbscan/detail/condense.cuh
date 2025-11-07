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
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#include <cstdint>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Condense {

template <typename value_idx, typename value_t>
value_t get_lambda_h(value_idx node, value_idx num_points, const value_t* deltas)
{
  value_t delta = deltas[node - num_points];
  return delta > 0.0 ? 1.0 / delta : std::numeric_limits<value_t>::max();
}

/*
 * Bottom-up implementation using omp threads
 * This is called from build_condensed_hierarchy
 */
template <typename value_idx, typename value_t>
void condense_hierarchy_cpu(const std::vector<value_idx>& children,
                            const std::vector<value_t>& deltas,
                            const std::vector<value_idx>& sizes,
                            std::vector<value_idx>& out_parent,
                            std::vector<value_idx>& out_child,
                            std::vector<value_t>& out_lambda,
                            std::vector<value_idx>& out_sizes,
                            int n_leaves,
                            int min_cluster_size)
{
  int n_nodes        = 2 * n_leaves - 1;
  int root           = n_nodes - 1;
  int internal_nodes = n_nodes - n_leaves;
  std::vector<int> parent(n_nodes, -1);
  std::vector<uint8_t> is_persistent(internal_nodes, false);

  /*
   * Mark whether each internal node forms a persistent cluster, and get parent information for easy
   * pointer chasing.
   *
   * A node is considered "persistent" if both of its child clusters contain at least
   * min_cluster_size points. Persistence indicates that this node can represent a valid cluster on
   * its own without either child being too small.
   */
#pragma omp parallel for
  for (int node = n_leaves; node < n_nodes; node++) {
    int left      = children[(node - n_leaves) * 2];
    int right     = children[(node - n_leaves) * 2 + 1];
    parent[left]  = node;
    parent[right] = node;

    int left_count  = (left >= n_leaves) ? sizes[left - n_leaves] : 1;
    int right_count = (right >= n_leaves) ? sizes[right - n_leaves] : 1;

    is_persistent[node - n_leaves] =
      (left_count >= min_cluster_size && right_count >= min_cluster_size) ? 1 : 0;
  }

  std::vector<int> relabel(n_nodes, 0);

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
    int cur             = node;
    int ancestor        = parent[cur];
    int rel             = -1;
    float assign_lambda = -1.0f;

    // climb until we reach root or a persistent parent
    while (ancestor != -1) {
      if (node < n_leaves && assign_lambda == -1 &&
          sizes[ancestor - n_leaves] >= min_cluster_size) {
        // if leaf node, keep track of delta value of the first internal node that has >= min
        // clusters
        assign_lambda = get_lambda_h(ancestor, n_leaves, deltas.data());
      }
      bool ancestor_persistent = (ancestor >= n_leaves && is_persistent[ancestor - n_leaves]);

      if (ancestor_persistent) {
        rel = cur;
        break;
      }

      cur      = ancestor;
      ancestor = parent[cur];
    }
    // anestor stays -1 for the root
    relabel[node] = ancestor == -1 ? root : rel;

    if (node < n_leaves) {
      out_parent[node * 2] = relabel[node];
      out_child[node * 2]  = node;
      out_lambda[node * 2] = assign_lambda;
      out_sizes[node * 2]  = 1;
    }
  }

  /*
   * Add edges for internal (persistent) nodes to their children using the relabel array.
   */
#pragma omp parallel for
  for (int node = n_leaves; node < n_nodes; node++) {
    if (is_persistent[node - n_leaves]) {
      value_idx left_child  = children[(node - n_leaves) * 2];
      value_idx right_child = children[((node - n_leaves) * 2) + 1];
      int left_count        = left_child >= n_leaves ? sizes[left_child - n_leaves] : 1;
      int right_count       = right_child >= n_leaves ? sizes[right_child - n_leaves] : 1;
      value_t lambda_value  = get_lambda_h(node, n_leaves, deltas.data());

      out_parent[node * 2] = relabel[node];
      out_child[node * 2]  = left_child;
      out_lambda[node * 2] = lambda_value;
      out_sizes[node * 2]  = left_count;

      out_parent[node * 2 + 1] = relabel[node];
      out_child[node * 2 + 1]  = right_child;
      out_lambda[node * 2 + 1] = lambda_value;
      out_sizes[node * 2 + 1]  = right_count;
    }
  }
}

/**
 * Condenses a binary single-linkage tree dendrogram in the Scipy hierarchy
 * format by collapsing subtrees that fall below a minimum cluster size.
 *
 * For increased parallelism, the output array sizes are held fixed but
 * the result will be sparse (e.g. zeros in place of parents who have been
 * removed / collapsed). This function accepts an empty instance of
 * `CondensedHierarchy` and invokes the `condense()` function on it to
 * convert the sparse output arrays into their dense form.
 *
 * @tparam value_idx
 * @tparam value_t
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
template <typename value_idx, typename value_t>
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

  value_idx root = 2 * (n_leaves - 1);

  // copying input arrays to host
  std::vector<value_idx> h_children(2 * (n_leaves - 1));
  std::vector<value_t> h_delta(n_leaves - 1);
  std::vector<value_idx> h_sizes(n_leaves - 1);
  raft::copy(h_children.data(), children, 2 * (n_leaves - 1), stream);
  raft::copy(h_delta.data(), delta, n_leaves - 1, stream);
  raft::copy(h_sizes.data(), sizes, n_leaves - 1, stream);
  handle.sync_stream(stream);

  // allocate output arrays
  std::vector<value_idx> h_out_parent((root + 1) * 2, -1);
  std::vector<value_idx> h_out_child((root + 1) * 2, -1);
  std::vector<value_t> h_out_lambda((root + 1) * 2, -1);
  std::vector<value_idx> h_out_sizes((root + 1) * 2, -1);

  condense_hierarchy_cpu(h_children,
                         h_delta,
                         h_sizes,
                         h_out_parent,
                         h_out_child,
                         h_out_lambda,
                         h_out_sizes,
                         n_leaves,
                         min_cluster_size);

  // allocate and copy output arrays to device
  rmm::device_uvector<value_idx> out_parent((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_child((root + 1) * 2, stream);
  rmm::device_uvector<value_t> out_lambda((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_size((root + 1) * 2, stream);
  raft::copy(out_parent.data(), h_out_parent.data(), (root + 1) * 2, stream);
  raft::copy(out_child.data(), h_out_child.data(), (root + 1) * 2, stream);
  raft::copy(out_lambda.data(), h_out_lambda.data(), (root + 1) * 2, stream);
  raft::copy(out_size.data(), h_out_sizes.data(), (root + 1) * 2, stream);
  handle.sync_stream(stream);

  condensed_tree.condense(out_parent.data(), out_child.data(), out_lambda.data(), out_size.data());
}

};  // end namespace Condense
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
