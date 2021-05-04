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

#include "membership.cuh"
#include "select.cuh"
#include "stabilities.cuh"
#include "utils.h"

#include <raft/sparse/op/sort.h>
#include <raft/sparse/convert/csr.cuh>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.cuh>

#include <algorithm>

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
namespace Extract {

template <typename value_idx>
class TreeUnionFind {
 public:
  TreeUnionFind(value_idx size) : data(size * 2, 0) {
    for (int i = 0; i < size; i++) {
      data[i * 2] = i;
    }
  }

  void perform_union(value_idx x, value_idx y) {
    value_idx x_root = find(x);
    value_idx y_root = find(y);

    if (data[x_root * 2 + 1] < data[y_root * 2 + 1])
      data[x_root * 2] = y_root;
    else if (data[x_root * 2 + 1] > data[y_root * 2 + 1])
      data[y_root * 2] = x_root;
    else {
      data[y_root * 2] = x_root;
      data[x_root * 2 + 1] += 1;
    }
  }

  value_idx find(value_idx x) {
    if (data[x * 2] != x) data[x * 2] = find(data[x * 2]);

    return data[x * 2];
  }

  value_idx *get_data() { return data.data(); }

 private:
  std::vector<value_idx> data;
};

template <typename value_idx, typename value_t>
void do_labelling_on_host(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  std::set<value_idx> &clusters, value_idx n_leaves, bool allow_single_cluster,
  value_idx *labels) {
  auto stream = handle.get_stream();

  std::vector<value_idx> children_h(condensed_tree.get_n_edges());
  std::vector<value_t> lambda_h(condensed_tree.get_n_edges());
  std::vector<value_idx> parent_h(condensed_tree.get_n_edges());

  raft::update_host(children_h.data(), condensed_tree.get_children(),
                    condensed_tree.get_n_edges(), stream);
  raft::update_host(parent_h.data(), condensed_tree.get_parents(),
                    condensed_tree.get_n_edges(), stream);
  raft::update_host(lambda_h.data(), condensed_tree.get_lambdas(),
                    condensed_tree.get_n_edges(), stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx size = *std::max_element(parent_h.begin(), parent_h.end());

  std::vector<value_idx> result(n_leaves);
  std::vector<value_t> parent_lambdas(size + 1, 0);

  auto union_find = TreeUnionFind<value_idx>(size + 1);

  for (int i = 0; i < condensed_tree.get_n_edges(); i++) {
    value_idx child = children_h[i];
    value_idx parent = parent_h[i];

    if (clusters.find(child) == clusters.end()) {
      union_find.perform_union(parent, child);
    }

    parent_lambdas[parent_h[i]] = max(parent_lambdas[parent_h[i]], lambda_h[i]);
  }

  for (int i = 0; i < n_leaves; i++) {
    value_idx cluster = union_find.find(i);

    if (cluster < n_leaves)
      result[i] = -1;
    else if (cluster == n_leaves) {
      //TODO: Implement the cluster_selection_epsilon / epsilon_search
      if (clusters.size() == 1 && allow_single_cluster) {
        auto it = std::find(children_h.begin(), children_h.end(), i);
        auto child_idx = std::distance(children_h.begin(), it);
        value_idx child_lambda = lambda_h[child_idx];
        if (child_lambda >= parent_lambdas[cluster])
          result[i] = cluster - n_leaves;
        else
          result[i] = -1;
      } else {
        result[i] = -1;
      }
    } else {
      result[i] = cluster - n_leaves;
    }
  }

  raft::update_device(labels, result.data(), n_leaves, stream);
}

/**
 * Extracts flattened labels using a cut point (epsilon) and minimum cluster
 * size. This is useful for Robust Single Linkage and DBSCAN labeling.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param children
 * @param deltas
 * @param sizes
 * @param n_leaves
 * @param cut
 * @param min_cluster_size
 * @param labels
 */
template <typename value_idx, typename value_t>
void do_labelling_at_cut(const raft::handle_t &handle,
                         const value_idx *children, const value_t *deltas,
                         value_idx n_leaves, double cut, int min_cluster_size,
                         value_idx *labels) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  // Root is the last edge in the dendrogram
  value_idx root = 2 * (n_leaves - 1);
  value_idx num_points = root / 2 + 1;

  std::vector<value_idx> labels_h(n_leaves);

  auto union_find = TreeUnionFind<value_idx>(root + 1);

  std::vector<value_idx> children_h(n_leaves * 2);
  std::vector<value_t> delta_h(n_leaves);

  raft::update_host(children_h.data(), children, n_leaves * 2, stream);
  raft::update_host(delta_h.data(), deltas, n_leaves, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx cluster = num_points;

  // Perform union on host to label parents / clusters
  for (int row = 0; row < n_leaves; row++) {
    if (delta_h[row] < cut) {
      union_find.perform_union(children_h[row * 2], cluster);
      union_find.perform_union(children_h[row * 2 + 1], cluster);
    }
    cluster += 1;
  }

  // Label points in parallel
  rmm::device_uvector<value_idx> union_find_data((root + 1) * 2, stream);
  raft::update_device(union_find_data.data(), union_find.get_data(),
                      union_find_data.size(), stream);

  rmm::device_uvector<value_idx> cluster_sizes(cluster, stream);
  thrust::fill(exec_policy, cluster_sizes.data(), cluster_sizes.data(), 0);

  auto seq = thrust::make_counting_iterator<value_idx>(0);

  value_idx *union_find_data_ptr = union_find_data.data();
  value_idx *cluster_sizes_ptr = cluster_sizes.data();

  thrust::for_each(exec_policy, seq, seq + n_leaves,
                   [=] __device__(value_idx leaf) {
                     // perform find using tree-union find
                     value_idx cur_find = union_find_data_ptr[leaf * 2];
                     while (cur_find != leaf)
                       cur_find = union_find_data_ptr[cur_find * 2];

                     labels[leaf] = cur_find;
                     atomicAdd(cluster_sizes_ptr + cur_find, 1);
                   });

  // Label noise points
  thrust::transform(exec_policy, labels, labels + n_leaves, labels,
                    [=] __device__(value_idx cluster) {
                      bool too_small =
                        cluster_sizes_ptr[cluster] < min_cluster_size;
                      return (too_small * -1) + (!too_small * cluster);
                    });

  // Draw non-noise points from a monotonically increasing set
  raft::label::make_monotonic(
    labels, labels, n_leaves, stream,
    [] __device__(value_idx label) { return label == -1; },
    handle.get_device_allocator(), true);
}

template <typename value_idx, typename value_t>
void extract_clusters(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  size_t n_leaves, value_idx *labels, value_t *stabilities,
  value_t *probabilities,
  Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
  bool allow_single_cluster = true, value_idx max_cluster_size = 0,
  value_t cluster_selection_epsilon = 0.0) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  rmm::device_uvector<value_t> tree_stabilities(condensed_tree.get_n_clusters(),
                                                handle.get_stream());

  Stability::compute_stabilities(handle, condensed_tree,
                                 tree_stabilities.data());

  rmm::device_uvector<int> is_cluster(condensed_tree.get_n_clusters(),
                                      handle.get_stream());

  if (max_cluster_size <= 0)
    max_cluster_size = n_leaves;  // negates the max cluster size

  auto cluster_tree = Utils::make_cluster_tree(handle, condensed_tree);

  if (cluster_selection_method == Common::CLUSTER_SELECTION_METHOD::EOM) {
    Select::excess_of_mass(handle, cluster_tree, tree_stabilities.data(),
                           is_cluster.data(), condensed_tree.get_n_clusters(),
                           max_cluster_size);
  } else if (cluster_selection_method ==
             Common::CLUSTER_SELECTION_METHOD::LEAF) {
    Select::leaf(handle, cluster_tree, is_cluster.data(),
                 condensed_tree.get_n_clusters());
  }

  if (cluster_selection_epsilon != 0.0) {
    auto epsilon_search = true;

    // this is to check when eom finds root as only cluster
    // in which case, epsilon search is cancelled
    if (cluster_selection_method == Common::CLUSTER_SELECTION_METHOD::EOM) {
      if (condensed_tree.get_n_clusters() == 1) {
        int is_root_only_cluster = false;
        raft::update_host(&is_root_only_cluster, is_cluster.data(), 1, stream);
        if (is_root_only_cluster) {
          epsilon_search = false;
        }
      }
    }

    if (epsilon_search) {
      Select::cluster_epsilon_search(handle, cluster_tree, is_cluster.data(),
                                     condensed_tree.get_n_clusters(),
                                     cluster_selection_epsilon,
                                     allow_single_cluster);
    }
  }

  std::vector<int> is_cluster_h(is_cluster.size());
  raft::update_host(is_cluster_h.data(), is_cluster.data(), is_cluster_h.size(),
                    stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::set<value_idx> clusters;
  for (int i = 0; i < is_cluster_h.size(); i++)
    if (is_cluster_h[i] != 0) clusters.insert(i + n_leaves);

  do_labelling_on_host<value_idx, value_t>(
    handle, condensed_tree, clusters, n_leaves, allow_single_cluster, labels);

  Membership::get_probabilities<value_idx, value_t>(handle, condensed_tree,
                                                    labels, probabilities);

  raft::label::make_monotonic(labels, labels, n_leaves, stream,
                              handle.get_device_allocator(), true);

  value_t max_lambda = *(thrust::max_element(
    exec_policy, condensed_tree.get_lambdas(),
    condensed_tree.get_lambdas() + condensed_tree.get_n_edges()));

  Stability::get_stability_scores(handle, labels, tree_stabilities.data(),
                                  clusters.size(), max_lambda, n_leaves,
                                  stabilities);
}

};  // end namespace Extract
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
