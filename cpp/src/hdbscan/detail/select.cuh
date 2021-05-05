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

#include <raft/sparse/op/sort.h>
#include <raft/sparse/convert/csr.cuh>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.cuh>

#include <algorithm>

#include "utils.h"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Select {

template <typename value_idx>
__global__ void propagate_cluster_negation_kernel(const value_idx *indptr,
                                                  const value_idx *children,
                                                  int *frontier,
                                                  int *is_cluster,
                                                  int n_clusters) {
  int cluster = blockDim.x * blockIdx.x + threadIdx.x;

  if (cluster < n_clusters && frontier[cluster]) {
    frontier[cluster] = false;

    value_idx children_start = indptr[cluster];
    value_idx children_stop = indptr[cluster + 1];
    for (int i = children_start; i < children_stop; i++) {
      value_idx child = children[i];
      frontier[child] = true;
      is_cluster[child] = false;
    }
  }
}

template <typename value_idx>
__global__ void get_cluster_tree_leaves(const value_idx *indptr,
                                        const value_idx *children,
                                        int *frontier, int *is_cluster,
                                        int n_clusters) {
  int cluster = blockDim.x * blockIdx.x + threadIdx.x;

  if (cluster < n_clusters && frontier[cluster]) {
    frontier[cluster] = false;

    value_idx children_start = indptr[cluster];
    value_idx children_stop = indptr[cluster + 1];

    if (children_stop - children_start == 0) {
      is_cluster[cluster] = true;
    }

    for (int i = children_start; i < children_stop; i++) {
      value_idx child = children[i];
      frontier[child] = true;
    }
  }
}

template <typename value_idx, typename Bfs_Kernel, int tpb = 256>
void perform_bfs(const raft::handle_t &handle, const value_idx *indptr,
                 const value_idx *children, int *frontier, int *is_cluster,
                 int n_clusters, Bfs_Kernel bfs_kernel) {
  auto stream = handle.get_stream();
  auto thrust_policy = rmm::exec_policy(stream);

  value_idx n_elements_to_traverse =
    thrust::reduce(thrust_policy, frontier, frontier + n_clusters, 0);

  // TODO: Investigate whether it's worth gathering the sparse frontier into
  // a dense form for purposes of uniform workload/thread scheduling

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(n_clusters, tpb);

  while (n_elements_to_traverse > 0) {
    bfs_kernel<<<grid, tpb, 0, stream>>>(indptr, children, frontier, is_cluster,
                                         n_clusters);

    n_elements_to_traverse =
      thrust::reduce(thrust_policy, frontier, frontier + n_clusters, 0);
  }
}

/**
  Function to get a csr index of parents of cluster tree.
  csr index is created by sorting parents by children then sizes
 */
template <typename value_idx, typename value_t>
void parent_csr(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &cluster_tree,
  const int n_clusters,
  value_idx *indptr) {
  auto stream = handle.get_stream();
  auto thrust_policy = rmm::exec_policy(stream);

  auto parents = cluster_tree.get_parents();
  auto children = cluster_tree.get_children();
  auto sizes = cluster_tree.get_sizes();
  auto cluster_tree_edges = cluster_tree.get_n_edges();

  raft::sparse::op::coo_sort(0, 0, cluster_tree_edges, parents, children, sizes,
                             handle.get_device_allocator(), stream);

  raft::sparse::convert::sorted_coo_to_csr(
    parents, cluster_tree_edges, indptr, n_clusters + 1,
    handle.get_device_allocator(), stream);
}

/**
 * Computes the excess of mass. This is a cluster extraction
 * strategy that iterates upwards from the leaves of the cluster
 * tree toward the root, selecting a cluster
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param handle
 * @param condensed_tree
 * @param stability an array of nodes from the cluster tree and their
 *                  corresponding stabilities
 * @param is_cluster
 * @param n_clusters
 * @param max_cluster_size
 */
template <typename value_idx, typename value_t, int tpb = 256>
void excess_of_mass(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &cluster_tree,
  value_t *stability,
  int *is_cluster,
  int n_clusters,
  value_idx max_cluster_size) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  /**
   * 1. Build CSR of cluster tree from condensed tree by filtering condensed tree for
   *    only those entries w/ lambda > 1 and constructing a CSR from the result
   */
  auto cluster_tree_edges = cluster_tree.get_n_edges();
  auto parents = cluster_tree.get_parents();
  auto children = cluster_tree.get_children();
  auto lambdas = cluster_tree.get_lambdas();
  auto sizes = cluster_tree.get_sizes();

  rmm::device_uvector<value_idx> cluster_sizes(n_clusters, stream);

  thrust::fill(exec_policy, cluster_sizes.data(),
               cluster_sizes.data() + cluster_sizes.size(), 0);

  value_idx *cluster_sizes_ptr = cluster_sizes.data();

  auto out =
    thrust::make_zip_iterator(thrust::make_tuple(parents, children, sizes));
  thrust::for_each(
    exec_policy, out, out + cluster_tree_edges,
    [=] __device__(const thrust::tuple<value_idx, value_idx, value_idx> &tup) {
      cluster_sizes_ptr[thrust::get<1>(tup)] = thrust::get<2>(tup);
    });

  /**
   * 2. Iterate through each level from leaves back to root. Use the cluster
   *    tree CSR and warp-level reduction to sum stabilities and test whether
   *    or not current cluster should continue to be its own
   */
  std::vector<int> is_cluster_h(n_clusters, true);
  std::vector<int> frontier_h(n_clusters, false);
  std::vector<value_idx> cluster_sizes_h(n_clusters);

  rmm::device_uvector<value_idx> indptr(n_clusters + 1, stream);
  parent_csr(handle, cluster_tree, n_clusters, indptr.data());

  std::vector<value_idx> indptr_h(indptr.size());
  raft::update_host(indptr_h.data(), indptr.data(), indptr.size(), stream);
  raft::update_host(cluster_sizes_h.data(), cluster_sizes.data(),
                    cluster_sizes.size(), stream);
  // don't need to sync here- thrust should take care of it.

  // Loop through stabilities in "reverse topological order" (e.g. reverse sorted order)
  for (value_idx node = n_clusters - 1; node >= 0; node--) {
    value_t node_stability = 0;
    raft::update_host(&node_stability, stability + node, 1, stream);

    value_t subtree_stability = 0;

    if (indptr_h[node + 1] - indptr_h[node] > 0) {
      subtree_stability = thrust::transform_reduce(
        exec_policy, children + indptr_h[node], children + indptr_h[node + 1],
        [=] __device__(value_idx a) { return stability[a]; }, 0,
        thrust::plus<value_t>());
    }

    if (subtree_stability > node_stability ||
        cluster_sizes_h[node] > max_cluster_size) {
      // Deselect / merge cluster with children
      raft::update_device(stability + node, &subtree_stability, 1, stream);
      is_cluster_h[node] = false;
    } else {
      // Mark children to be deselected
      frontier_h[node] = true;
    }
  }

  /**
   * 3. Perform BFS through is_cluster, propagating cluster
   * "deselection" through subtrees
   */
  rmm::device_uvector<int> cluster_propagate(n_clusters, stream);
  rmm::device_uvector<int> frontier(n_clusters, stream);

  raft::update_device(is_cluster, is_cluster_h.data(), n_clusters, stream);
  raft::update_device(frontier.data(), frontier_h.data(), n_clusters, stream);

  perform_bfs(handle, indptr.data(), children, frontier.data(), is_cluster,
              n_clusters, propagate_cluster_negation_kernel<value_idx>);
}

/**
 * Computes cluster selection using leaf method
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param handle
 * @param condensed_tree
 * @param is_cluster
 * @param n_clusters
 */
template <typename value_idx, typename value_t, int tpb = 256>
void leaf(const raft::handle_t &handle,
          Common::CondensedHierarchy<value_idx, value_t> &cluster_tree,
          int *is_cluster, int n_clusters) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  auto children = cluster_tree.get_children();

  rmm::device_uvector<value_idx> indptr(n_clusters + 1, stream);
  parent_csr(handle, cluster_tree, n_clusters, indptr.data());
  thrust::fill(exec_policy, is_cluster, is_cluster + n_clusters, false);

  // mark root in frontier
  rmm::device_uvector<int> frontier(n_clusters, stream);
  constexpr int root_in_fontier = true;
  frontier.set_element(0, root_in_fontier, stream);

  perform_bfs(handle, indptr.data(), children, frontier.data(), is_cluster,
              n_clusters, get_cluster_tree_leaves<value_idx>);

  auto n_selected_clusters =
    thrust::reduce(exec_policy, is_cluster, is_cluster + n_clusters);

  // if no cluster leaves were found, declare root as cluster
  if (n_selected_clusters == 0) {
    int root_is_cluster = true;
    raft::update_device(is_cluster, &root_is_cluster, 1, stream);
  }
  raft::print_device_vector("is_cluster_leaf", is_cluster, n_clusters,
                            std::cout);
}

template <typename value_idx, typename value_t, int tpb = 256>
__global__ void cluster_epsilon_search_kernel(
  const int *selected_clusters, const int n_selected_clusters,
  const value_idx *parents, const value_idx *children, const value_t *lambdas,
  const value_idx cluster_tree_edges, int *is_cluster, int *frontier,
  const int n_clusters, const value_t cluster_selection_epsilon,
  const bool allow_single_cluster) {
  auto selected_cluster_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (selected_cluster_idx >= n_selected_clusters) {
    return;
  }

  // subtract 1 to offset for root
  // this is because children array will never have a value of 0
  // this works because children array is sorted
  auto child_idx = selected_clusters[selected_cluster_idx] - 1;

  auto eps = 1 / lambdas[child_idx];

  if (eps < cluster_selection_epsilon) {
    constexpr auto root = 0;
    auto parent = parents[child_idx];

    value_t parent_eps;

    do {
      if (parent == root) {
        if (!allow_single_cluster) {
          parent = child_idx + 1;
        }
        break;
      }

      // again offsetting for root
      child_idx = parent - 1;
      parent = parents[child_idx];
      parent_eps = 1 / lambdas[child_idx];
    } while (parent_eps <= cluster_selection_epsilon);

    frontier[parent] = true;
    is_cluster[parent] = true;
  } else {
    frontier[child_idx + 1] = true;
  }
}

template <typename value_idx, typename value_t, int tpb = 256>
void cluster_epsilon_search(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &cluster_tree, int *is_cluster,
  const value_idx n_clusters, const value_t cluster_selection_epsilon,
  const bool allow_single_cluster) {
  auto stream = handle.get_stream();
  auto thrust_policy = rmm::exec_policy(stream);
  auto parents = cluster_tree.get_parents();
  auto children = cluster_tree.get_children();
  auto lambdas = cluster_tree.get_lambdas();
  auto cluster_tree_edges = cluster_tree.get_n_edges();

  auto n_selected_clusters =
    thrust::reduce(thrust_policy, is_cluster, is_cluster + n_clusters);

  rmm::device_uvector<int> selected_clusters(n_selected_clusters, stream);

  // copying selected clusters by index
  thrust::copy_if(thrust_policy, thrust::make_counting_iterator(value_idx(0)),
                  thrust::make_counting_iterator(n_clusters), is_cluster,
                  selected_clusters.data(),
                  [] __device__(auto cluster) { return cluster; });

  // sort lambdas and parents by children for epsilon search
  auto start = thrust::make_zip_iterator(thrust::make_tuple(parents, lambdas));
  thrust::sort_by_key(thrust_policy, children, children + cluster_tree_edges,
                      start);

  // declare frontier and search
  rmm::device_uvector<int> frontier(n_clusters, stream);
  thrust::fill(thrust_policy, frontier.begin(), frontier.end(), false);

  auto nblocks = raft::ceildiv(n_selected_clusters, tpb);
  cluster_epsilon_search_kernel<<<nblocks, tpb, 0, stream>>>(
    selected_clusters.data(), n_selected_clusters, parents, children, lambdas,
    cluster_tree_edges, is_cluster, frontier.data(), n_clusters,
    cluster_selection_epsilon, allow_single_cluster);

  rmm::device_uvector<value_idx> indptr(n_clusters + 1, stream);
  parent_csr(handle, cluster_tree, n_clusters, indptr.data());

  perform_bfs(handle, indptr.data(), children, frontier.data(), is_cluster,
              n_clusters, propagate_cluster_negation_kernel<value_idx>);

}

template<typename value_idx, typename value_t>
void select_clusters(const raft::handle_t &handle,
                     Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
                     value_t *tree_stabilities,
                     int *is_cluster,
                     Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
                     bool allow_single_cluster,
                     int max_cluster_size,
                     float cluster_selection_epsilon) {

  auto stream = handle.get_stream();

  CUML_LOG_DEBUG("Building cluster tree: n_clusters=%d", condensed_tree.get_n_clusters());
  auto cluster_tree = Utils::make_cluster_tree(handle, condensed_tree);

  if (cluster_selection_method == Common::CLUSTER_SELECTION_METHOD::EOM) {
    Select::excess_of_mass(handle, cluster_tree, tree_stabilities,
                           is_cluster, condensed_tree.get_n_clusters(),
                           max_cluster_size);
  } else {
    leaf(handle, cluster_tree, is_cluster,
         condensed_tree.get_n_clusters());
  }

  if (cluster_selection_epsilon != 0.0) {
    auto epsilon_search = true;

    // this is to check when eom finds root as only cluster
    // in which case, epsilon search is cancelled
    if (cluster_selection_method == Common::CLUSTER_SELECTION_METHOD::EOM) {
      if (condensed_tree.get_n_clusters() == 1) {
        int is_root_only_cluster = false;
        raft::update_host(&is_root_only_cluster, is_cluster, 1, stream);
        if (is_root_only_cluster) {
          epsilon_search = false;
        }
      }
    }

    if (epsilon_search) {
      Select::cluster_epsilon_search(handle, cluster_tree, is_cluster,
                                     condensed_tree.get_n_clusters(),
                                     cluster_selection_epsilon,
                                     allow_single_cluster);
    }
  }

}

};  // namespace Select
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
