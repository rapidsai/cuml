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

template <typename value_idx, typename value_t>
void condense_hierarchy_pointer_chase(
  const std::vector<value_idx>& children,  // 2*(n_leaves-1) array 32
  const std::vector<value_t>& deltas,      // (n_leaves-1) array 16
  const std::vector<value_idx>& sizes,     // (n_leaves-1) array 16
  std::vector<value_idx>& out_parent,      // (root + 1) * 2 66
  std::vector<value_idx>& out_child,
  std::vector<value_t>& out_lambda,
  std::vector<value_idx>& out_sizes,
  int n_leaves,         // 17
  int min_cluster_size  // 3
)
{
  int n_nodes        = 2 * n_leaves - 1;    // 33
  int root           = n_nodes - 1;         // 32
  int internal_nodes = n_nodes - n_leaves;  // 16 (17 ~ 32)
  std::vector<int> parent(n_nodes, -1);     // root node has parent -1
  std::vector<int> is_persistent(internal_nodes, false);

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

  // find your relabel
  std::vector<int> relabel(n_nodes, 0);
  // only this part really needs to be done on the cpu
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
// template <typename value_idx, typename value_t, int tpb = 256>
// void build_condensed_hierarchy(const raft::handle_t& handle,
//                                const value_idx* children,
//                                const value_t* delta,
//                                const value_idx* sizes,
//                                int min_cluster_size,
//                                int n_leaves,
//                                Common::CondensedHierarchy<value_idx, value_t>& condensed_tree)
// {
//   cudaStream_t stream = handle.get_stream();
//   auto exec_policy    = handle.get_thrust_policy();

//   // Root is the last edge in the dendrogram
//   value_idx root = 2 * (n_leaves - 1);

//   auto d_ptr           = thrust::device_pointer_cast(children);
//   value_idx n_vertices = *(thrust::max_element(exec_policy, d_ptr, d_ptr + root)) + 1;

//   // Prevent potential infinite loop from labeling disconnected
//   // connectivities graph.
//   RAFT_EXPECTS(n_vertices == root,
//                "Multiple components found in MST or MST is invalid. "
//                "Cannot find single-linkage solution. Found %d vertices "
//                "total.",
//                static_cast<int>(n_vertices));

//   rmm::device_uvector<bool> frontier(root + 1, stream);
//   rmm::device_uvector<bool> next_frontier(root + 1, stream);

//   thrust::fill(exec_policy, frontier.begin(), frontier.end(), false);
//   thrust::fill(exec_policy, next_frontier.begin(), next_frontier.end(), false);

//   // Array to propagate the lambda of subtrees actively being collapsed
//   // through multiple bfs iterations.
//   rmm::device_uvector<value_t> ignore(root + 1, stream);

//   // Propagate labels from root
//   rmm::device_uvector<value_idx> relabel(root + 1, handle.get_stream());
//   thrust::fill(exec_policy, relabel.begin(), relabel.end(), -1);

//   raft::update_device(relabel.data() + root, &root, 1, handle.get_stream());

//   // Flip frontier for root
//   constexpr bool start = true;
//   raft::update_device(frontier.data() + root, &start, 1, handle.get_stream());

//   rmm::device_uvector<value_idx> out_parent((root + 1) * 2, stream);
//   rmm::device_uvector<value_idx> out_child((root + 1) * 2, stream);
//   rmm::device_uvector<value_t> out_lambda((root + 1) * 2, stream);
//   rmm::device_uvector<value_idx> out_size((root + 1) * 2, stream);

//   thrust::fill(exec_policy, out_parent.begin(), out_parent.end(), -1);
//   thrust::fill(exec_policy, out_child.begin(), out_child.end(), -1);
//   thrust::fill(exec_policy, out_lambda.begin(), out_lambda.end(), -1);
//   thrust::fill(exec_policy, out_size.begin(), out_size.end(), -1);
//   thrust::fill(exec_policy, ignore.begin(), ignore.end(), -1);

//   // While frontier is not empty, perform single bfs through tree
//   size_t grid = raft::ceildiv(root + 1, static_cast<value_idx>(tpb));

//   value_idx n_elements_to_traverse =
//     thrust::reduce(exec_policy, frontier.data(), frontier.data() + root + 1, 0);

//   while (n_elements_to_traverse > 0) {
//     // TODO: Investigate whether it would be worth performing a gather/argmatch in order
//     // to schedule only the number of threads needed. (it might not be worth it)
//     condense_hierarchy_kernel<<<grid, tpb, 0, handle.get_stream()>>>(frontier.data(),
//                                                                      next_frontier.data(),
//                                                                      ignore.data(),
//                                                                      relabel.data(),
//                                                                      children,
//                                                                      delta,
//                                                                      sizes,
//                                                                      n_leaves,
//                                                                      min_cluster_size,
//                                                                      out_parent.data(),
//                                                                      out_child.data(),
//                                                                      out_lambda.data(),
//                                                                      out_size.data());

//     thrust::copy(exec_policy, next_frontier.begin(), next_frontier.end(), frontier.begin());
//     thrust::fill(exec_policy, next_frontier.begin(), next_frontier.end(), false);

//     n_elements_to_traverse = thrust::reduce(
//       exec_policy, frontier.data(), frontier.data() + root + 1, static_cast<value_idx>(0));

//     handle.sync_stream(stream);
//   }

//   condensed_tree.condense(out_parent.data(), out_child.data(), out_lambda.data(),
//   out_size.data());
// }

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

  value_idx root = 2 * (n_leaves - 1);

  std::vector<value_idx> h_children(2 * (n_leaves - 1));
  std::vector<value_t> h_delta(n_leaves - 1);
  std::vector<value_idx> h_sizes(n_leaves - 1);

  raft::copy(h_children.data(), children, 2 * (n_leaves - 1), stream);
  raft::copy(h_delta.data(), delta, n_leaves - 1, stream);
  raft::copy(h_sizes.data(), sizes, n_leaves - 1, stream);
  handle.sync_stream(stream);

  std::vector<value_idx> h_out_parent((root + 1) * 2, -1);
  std::vector<value_idx> h_out_child((root + 1) * 2, -1);
  std::vector<value_t> h_out_lambda((root + 1) * 2, -1);
  std::vector<value_idx> h_out_sizes((root + 1) * 2, -1);

  condense_hierarchy_pointer_chase(h_children,
                                   h_delta,
                                   h_sizes,
                                   h_out_parent,
                                   h_out_child,
                                   h_out_lambda,
                                   h_out_sizes,
                                   n_leaves,
                                   min_cluster_size);

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
