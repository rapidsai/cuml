/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "kernels/stabilities.cuh"
#include "utils.h"

#include <cub/cub.cuh>

#include <raft/cudart_utils.h>

#include <raft/sparse/convert/csr.hpp>
#include <raft/sparse/op/sort.hpp>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.hpp>

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
namespace Stability {

/**
 * Computes stability scores which are used for excess of mass cluster
 * selection. Stabilities are computed over the points in each cluster as the sum
 * of the lambda (1 / distance) of each point minus the lambda of its parent.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree condensed hierarchy (size n_points + n_clusters)
 * @param[out] stabilities output stabilities array (size n_clusters)
 */
template <typename value_idx, typename value_t>
void compute_stabilities(const raft::handle_t& handle,
                         Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                         value_t* stabilities)
{
  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto lambdas    = condensed_tree.get_lambdas();
  auto sizes      = condensed_tree.get_sizes();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();

  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  rmm::device_uvector<value_idx> sorted_parents(n_edges, stream);
  raft::copy_async(sorted_parents.data(), parents, n_edges, stream);

  rmm::device_uvector<value_idx> sorted_parents_offsets(n_edges + 1, stream);
  Utils::parent_csr(handle, condensed_tree, sorted_parents.data(), sorted_parents_offsets.data());

  // This is to consider the case where a child may also be a parent
  // in which case, births for that parent are initialized to
  // lambda for that child
  rmm::device_uvector<value_t> births(n_clusters, stream);
  thrust::fill(exec_policy, births.begin(), births.end(), 0.0f);
  auto births_init_op =
    [n_leaves, children, lambdas, births = births.data()] __device__(const auto& idx) {
      auto child = children[idx];
      if (child >= n_leaves) { births[child - n_leaves] = lambdas[idx]; }
    };

  // this is to find minimum lambdas of all children under a parent
  rmm::device_uvector<value_t> births_parent_min(n_clusters, stream);
  thrust::for_each(exec_policy,
                   thrust::make_counting_iterator(value_idx(0)),
                   thrust::make_counting_iterator(n_edges),
                   births_init_op);

  Utils::cub_segmented_reduce(
    lambdas,
    births_parent_min.data() + 1,
    n_clusters - 1,
    sorted_parents_offsets.data() + 1,
    stream,
    cub::DeviceSegmentedReduce::Min<const value_t*, value_t*, const value_idx*, const value_idx*>);
  // finally, we find minimum between initialized births where parent=child
  // and births of parents for their childrens
  auto births_zip =
    thrust::make_zip_iterator(thrust::make_tuple(births.data(), births_parent_min.data()));
  auto min_op = [] __device__(const thrust::tuple<value_t, value_t>& birth_pair) {
    auto birth             = thrust::get<0>(birth_pair);
    auto births_parent_min = thrust::get<1>(birth_pair);

    return birth < births_parent_min ? birth : births_parent_min;
  };
  thrust::transform(
    exec_policy, births_zip + 1, births_zip + n_clusters, births.begin() + 1, min_op);

  thrust::fill(exec_policy, stabilities, stabilities + n_clusters, 0.0f);

  // for each child, calculate summation (lambda[child] - birth[parent]) * sizes[child]
  stabilities_functor<value_idx, value_t> stabilities_op(
    stabilities, births.data(), parents, lambdas, sizes, n_leaves);
  thrust::for_each(exec_policy,
                   thrust::make_counting_iterator(value_idx(0)),
                   thrust::make_counting_iterator(n_edges),
                   stabilities_op);
}

/**
 * Computes stability scores for each cluster by normalizing their
 * stabilities by their sizes and scaling by the lambda of the root.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] labels labels array (size n_leaves)
 * @param[in] stability stabilities array (size n_clusters)
 * @param[in] n_condensed_clusters number of clusters in cluster tree
 * @param[in] max_lambda maximum lambda of cluster hierarchy
 * @param[in] n_leaves number of data points (non-clusters) in hierarchy
 * @param[out] result output stability scores
 * @param[in] label_map map of original labels to new final labels (size n_leaves)
 */
template <typename value_idx, typename value_t>
void get_stability_scores(const raft::handle_t& handle,
                          const value_idx* labels,
                          const value_t* stability,
                          size_t n_condensed_clusters,
                          value_t max_lambda,
                          size_t n_leaves,
                          value_t* result,
                          value_idx* label_map)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  /**
   * 1. Populate cluster sizes
   */
  rmm::device_uvector<value_idx> cluster_sizes(n_leaves, handle.get_stream());
  thrust::fill(exec_policy, cluster_sizes.data(), cluster_sizes.data() + cluster_sizes.size(), 0);

  value_idx* sizes = cluster_sizes.data();
  thrust::for_each(exec_policy, labels, labels + n_leaves, [=] __device__(value_idx v) {
    if (v > -1) atomicAdd(sizes + v, 1);
  });

  /**
   * Compute stability scores
   */

  auto enumeration = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator(0), cluster_sizes.data()));
  thrust::for_each(exec_policy,
                   enumeration,
                   enumeration + n_condensed_clusters,
                   [=] __device__(thrust::tuple<value_idx, value_idx> tup) {
                     value_idx size        = thrust::get<1>(tup);
                     value_idx c           = thrust::get<0>(tup);
                     value_idx out_cluster = label_map[c];

                     if (out_cluster >= 0) {
                       bool expr = max_lambda == std::numeric_limits<value_t>::max() ||
                                   max_lambda == 0.0 || size == 0;
                       if (expr)
                         result[out_cluster] = 1.0f;
                       else
                         result[out_cluster] = stability[c] / (size * max_lambda);
                     }
                   });
}

};  // namespace Stability
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
