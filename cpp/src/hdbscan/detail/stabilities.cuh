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

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include "utils.h"

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Stability {

template <typename value_idx, typename value_t>
struct stabilities_functor {
 public:
  stabilities_functor(value_t *stabilities_, const value_t *births_,
                      const value_idx *parents_, const value_idx *children_,
                      const value_t *lambdas_, const value_idx *sizes_,
                      const value_idx n_leaves_)
    : stabilities(stabilities_),
      births(births_),
      parents(parents_),
      children(children_),
      lambdas(lambdas_),
      sizes(sizes_),
      n_leaves(n_leaves_) {}

  __device__ void operator()(const int &idx) {
    auto parent = parents[idx] - n_leaves;

    atomicAdd(&stabilities[parent],
              (lambdas[idx] - births[parent]) * sizes[idx]);
  }

 private:
  value_t *stabilities;
  const value_t *births, *lambdas;
  const value_idx *parents, *children, *sizes, n_leaves;
};

template <typename value_idx, typename value_t>
void compute_stabilities(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  value_t *stabilities) {
  auto parents = condensed_tree.get_parents();
  auto children = condensed_tree.get_children();
  auto lambdas = condensed_tree.get_lambdas();
  auto sizes = condensed_tree.get_sizes();
  auto n_edges = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves = condensed_tree.get_n_leaves();

  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  // TODO: Reverse topological sort (e.g. sort hierarchy, lambdas, and sizes by lambda)
  rmm::device_uvector<value_t> sorted_lambdas(n_edges, stream);
  raft::copy_async(sorted_lambdas.data(), lambdas, n_edges, stream);

  rmm::device_uvector<value_idx> sorted_parents(n_edges, stream);
  raft::copy_async(sorted_parents.data(), parents, n_edges, stream);

  thrust::sort_by_key(exec_policy, sorted_parents.begin(), sorted_parents.end(),
                      sorted_lambdas.begin());

  // 0-index sorted parents by subtracting n_leaves for offsets and birth/stability indexing
  auto index_op = [n_leaves] __device__(const auto &x) { return x - n_leaves; };
  thrust::transform(exec_policy, sorted_parents.begin(), sorted_parents.end(),
                    sorted_parents.begin(), index_op);

  rmm::device_uvector<value_idx> sorted_parents_offsets(n_edges + 1, stream);
  raft::sparse::convert::sorted_coo_to_csr(
    sorted_parents.data(), n_edges, sorted_parents_offsets.data(),
    n_clusters + 1, handle.get_device_allocator(), handle.get_stream());

  // Segmented reduction on min_lambda within each cluster
  // TODO: Converting child array to CSR offset and using CUB Segmented Reduce
  // Investigate use of a kernel like coo_spmv

  // This is to consider the case where a child may also be a parent
  // in which case, births for that parent are initialized to
  // lambda for that child
  rmm::device_uvector<value_t> births(n_clusters, stream);
  thrust::fill(exec_policy, births.begin(), births.end(), 0.0f);
  auto births_init_op = [n_leaves, children, lambdas,
                         births = births.data()] __device__(const auto &idx) {
    auto child = children[idx];
    if (child >= n_leaves) {
      births[child - n_leaves] = lambdas[idx];
    }
  };
  raft::print_device_vector("inital_births", births.data(), n_clusters, std::cout);

  // this is to find minimum lambdas of all children under a prent
  rmm::device_uvector<value_t> births_parent_min(n_clusters, stream);
  thrust::fill(exec_policy, births.begin(), births.end(), 0.0f);
  thrust::for_each(exec_policy, thrust::make_counting_iterator(value_idx(0)),
                   thrust::make_counting_iterator(n_edges), births_init_op);
  Utils::segmented_reduce(
    sorted_lambdas.data(), births_parent_min.data() + 1, n_clusters - 1,
    sorted_parents_offsets.data() + 1, stream,
    cub::DeviceSegmentedReduce::Min<const value_t *, value_t *,
                                    const value_idx *>);
  raft::print_device_vector("min_births", births_parent_min.data(), n_clusters, std::cout);
  // finally, we find minimum between initialized births where parent=child
  // and births of parents for their childrens
  auto births_zip = thrust::make_zip_iterator(
    thrust::make_tuple(births.data(), births_parent_min.data()));
  auto min_op =
    [] __device__(const thrust::tuple<value_t, value_t> &birth_pair) {
      auto birth = thrust::get<0>(birth_pair);
      auto births_parent_min = thrust::get<1>(birth_pair);

      return birth < births_parent_min ? birth : births_parent_min;
    };
  thrust::transform(exec_policy, births_zip + 1, births_zip + n_clusters,
                    births.begin() + 1, min_op);

  raft::print_device_vector("Final births", births.data(), n_clusters, std::cout);

  thrust::fill(exec_policy, stabilities, stabilities + n_clusters, 0.0f);

  // for each child, calculate summation (lambda[child] - lambda[birth[parent]]) * sizes[child]
  stabilities_functor<value_idx, value_t> stabilities_op(
    stabilities, births.data(), parents, children, lambdas, sizes, n_leaves);
  thrust::for_each(exec_policy, thrust::make_counting_iterator(value_idx(0)),
                   thrust::make_counting_iterator(n_edges), stabilities_op);

  raft::print_device_vector("stabilities", stabilities, n_clusters, std::cout);
}

template <typename value_idx, typename value_t>
void get_stability_scores(const raft::handle_t &handle, const value_idx *labels,
                          const value_t *stability, size_t n_clusters,
                          value_t max_lambda, size_t n_leaves,
                          value_t *result) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  /**
   * 1. Populate cluster sizes
   */
  rmm::device_uvector<value_idx> cluster_sizes(n_clusters, handle.get_stream());
  thrust::fill(exec_policy, cluster_sizes.data(), cluster_sizes.data()+cluster_sizes.size(), 0);

  value_idx *sizes = cluster_sizes.data();
  thrust::for_each(exec_policy, labels, labels + n_leaves,
                   [=] __device__(value_idx v) {
                     if(v > -1)
                       atomicAdd(sizes + v, 1);
                   });

  /**
   * Compute stability scores
   */
  auto enumeration = thrust::make_zip_iterator(thrust::make_tuple(
    thrust::make_counting_iterator(0), cluster_sizes.data()));
  thrust::transform(
    exec_policy, enumeration, enumeration + n_clusters, result,
    [=] __device__(thrust::tuple<value_idx, value_idx> tup) {
      value_idx size = thrust::get<1>(tup);
      value_idx c = thrust::get<0>(tup);

      bool expr = max_lambda == std::numeric_limits<value_t>::max() ||
                  max_lambda == 0.0 || size == 0;
      return (!expr * (stability[c] / size * max_lambda)) + (expr * 1.0);
    });
}

};  // namespace Stability
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML