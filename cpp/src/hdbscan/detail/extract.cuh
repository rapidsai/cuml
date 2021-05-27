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

/**
 * Union-rank data structure with path compression for
 * labeling data points based on their farthest ancestors
 * under root.
 *
 * For correctness, it is important that all children are
 * visited before their parents.
 * @tparam value_idx
 */
template <typename value_idx>
class TreeUnionFind {
 public:
  TreeUnionFind(value_idx size_) : size(size_), data(size_ * 2, 0) {
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
    if (data[x * 2] != x) {
      data[x * 2] = find(data[x * 2]);
    }

    return data[x * 2];
  }

  value_idx *get_data() { return data.data(); }

 private:
  value_idx size;
  std::vector<value_idx> data;
};

template <typename value_idx, typename value_t>
void do_labelling_on_host(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  std::set<value_idx> &clusters, value_idx n_leaves, bool allow_single_cluster,
  value_idx *labels, value_t cluster_selection_epsilon) {
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

  auto parents = thrust::device_pointer_cast(condensed_tree.get_parents());
  auto thrust_policy = rmm::exec_policy(stream);
  value_idx size = *thrust::max_element(thrust_policy, parents,
                                        parents + condensed_tree.get_n_edges());

  std::vector<value_idx> result(n_leaves);
  std::vector<value_t> parent_lambdas(size + 1, 0);

  auto union_find = TreeUnionFind<value_idx>(size + 1);

  for (int i = 0; i < condensed_tree.get_n_edges(); i++) {
    value_idx child = children_h[i];
    value_idx parent = parent_h[i];

    if (clusters.find(child) == clusters.end())
      union_find.perform_union(parent, child);

    parent_lambdas[parent_h[i]] = max(parent_lambdas[parent_h[i]], lambda_h[i]);
  }

  value_t inverse_cluster_selection_epsilon;
  if (cluster_selection_epsilon != 0.0) {
    inverse_cluster_selection_epsilon = 1 / cluster_selection_epsilon;
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
        value_t child_lambda = lambda_h[child_idx];

        if (cluster_selection_epsilon != 0) {
          if (child_lambda >= inverse_cluster_selection_epsilon) {
            result[i] = cluster - n_leaves;
          } else {
            result[i] = -1;
          }
        } else if (child_lambda >= parent_lambdas[cluster])
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

  // TODO: Need to nornalize the labels so they are pulled from a
  // monotonically increasing set, but need to make sure the first
  // label starts at 0 in the face of noise.
  //  CUML_LOG_DEBUG("Calling make_monotonic");
  //  raft::label::make_monotonic(labels, labels, n_leaves, stream,
  //                              [] __device__(value_idx label) { return label == -1; },
  //                              handle.get_device_allocator(), true);
}

/**
 * Compute cluster stabilities, perform cluster selection, and
 * label the resulting clusters. In addition, probabilities
 * are computed and stabilities are normalized into scores.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource management
 * @param[in] condensed_tree a condensed hierarchy
 * @param[in] n_leaves number of data samples
 * @param[out] labels array of labels on device (size n_leaves)
 * @param[out] stabilities array of stabilities on device (size n_clusters)
 * @param[out] probabilities array of probabilities on device (size n_leaves)
 * @param[in] cluster_selection_method method to use for cluster selection
 * @param[in] allow_single_cluster allows a single cluster to be returned (rather than just noise)
 * @param[in] max_cluster_size maximium number of points that can be considered in a cluster before it is split into multiple sub-clusters.
 * @param[in] cluster_selection_epsilon a distance threshold. clusters below this value will be merged.
 */
template <typename value_idx, typename value_t>
void extract_clusters(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  size_t n_leaves, value_idx *labels, value_t *stabilities,
  value_t *probabilities,
  Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
  bool allow_single_cluster = false, value_idx max_cluster_size = 0,
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

  Select::select_clusters(handle, condensed_tree, tree_stabilities.data(),
                          is_cluster.data(), cluster_selection_method,
                          allow_single_cluster, max_cluster_size,
                          cluster_selection_epsilon);

  std::vector<int> is_cluster_h(is_cluster.size());
  raft::update_host(is_cluster_h.data(), is_cluster.data(), is_cluster_h.size(),
                    stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::set<value_idx> clusters;
  for (int i = 0; i < is_cluster_h.size(); i++) {
    if (is_cluster_h[i] != 0) {
      clusters.insert(i + n_leaves);
    }
  }

  do_labelling_on_host<value_idx, value_t>(handle, condensed_tree, clusters,
                                           n_leaves, allow_single_cluster,
                                           labels, cluster_selection_epsilon);

  value_idx n_selected_clusters = clusters.size();

  Membership::get_probabilities<value_idx, value_t>(handle, condensed_tree,
                                                    labels, probabilities);

  // TODO: Compute stability scores (below)

  //  auto lambdas_ptr = thrust::device_pointer_cast(condensed_tree.get_lambdas());
  //  value_t max_lambda = *(thrust::max_element(
  //    exec_policy, lambdas_ptr,
  //    lambdas_ptr + condensed_tree.get_n_edges()));
  //
  //  CUML_LOG_DEBUG("Computing stability scores");
  //  Stability::get_stability_scores(handle, labels, tree_stabilities.data(),
  //                                  clusters.size(), max_lambda, n_leaves,
  //                                  stabilities);
}

};  // end namespace Extract
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
