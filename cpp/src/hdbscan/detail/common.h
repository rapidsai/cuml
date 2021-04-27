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

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Common {

template <typename value_idx, typename value_t>
struct CondensedHierarchy {
  CondensedHierarchy(const raft::handle_t &handle_, size_t n_leaves_)
    : handle(handle_),
      n_leaves(n_leaves_),
      parents(0, handle.get_stream()),
      children(0, handle.get_stream()),
      lambdas(0, handle.get_stream()),
      sizes(0, handle.get_stream()) {}

  /**
   * Populates the condensed hierarchy object with the output
   * from Condense::condense_hierarchy
   * @param full_parents
   * @param full_children
   * @param full_lambdas
   * @param full_sizes
   */
  void condense(value_idx *full_parents, value_idx *full_children,
                value_t *full_lambdas, value_idx *full_sizes) {
    auto stream = handle.get_stream();

    n_edges = thrust::transform_reduce(
      thrust::cuda::par.on(stream), full_sizes, full_sizes + (4 * (n_leaves - 1) + 2),
      [=] __device__(value_idx a) { return a != -1; }, 0,
      thrust::plus<value_idx>());

    parents.resize(n_edges, stream);
    children.resize(n_edges, stream);
    lambdas.resize(n_edges, stream);
    sizes.resize(n_edges, stream);

    auto in = thrust::make_zip_iterator(thrust::make_tuple(
      full_parents, full_children, full_lambdas, full_sizes));

    auto out = thrust::make_zip_iterator(thrust::make_tuple(
      parents.data(), children.data(), lambdas.data(), sizes.data()));

    thrust::copy_if(
      thrust::cuda::par.on(stream), in, in + (4 * (n_leaves - 1) + 2), out,
      [=] __device__(
        thrust::tuple<value_idx, value_idx, value_t, value_idx> tup) {
        return thrust::get<3>(tup) != -1;
      });

    // TODO: I don't believe this is correct. The whole set of
    //  parents/children will need to be made monotonic.
    // Also, make_monotonic doesn't have a return value.
    //    n_clusters = MLCommon::Label::make_monotonic(
    //      handle, parents.data(), parents.begin(), parents.end());
  }

  /**
   * Builds a cluster tree by filtering out the leaves (data samples)
   * of the condensed hierarchy, keeping only the internal cluster
   * tree.
   */
  void cluster_tree() {
    // TODO: Move the code from step #1 in `excess_of_mass` into here.
    // get_cluster_tree_edges() can be used to compute the size of
    // the resulting cluster tree
  }

  value_idx get_cluster_tree_edges() {
    return thrust::transform_reduce(
      thrust::cuda::par.on(handle.get_stream()), get_sizes(),
      get_sizes() + get_n_edges(),
      [=] __device__(value_t a) { return a > 1; }, 0,
      thrust::plus<value_idx>());
  }

  value_idx *get_parents() { return parents.data(); }

  value_idx *get_children() { return children.data(); }

  value_t *get_lambdas() { return lambdas.data(); }

  value_idx *get_sizes() { return sizes.data(); }

  value_idx get_n_edges() { return n_edges; }

  int get_n_clusters() { return n_clusters; }

 private:
  const raft::handle_t &handle;

  rmm::device_uvector<value_idx> parents;
  rmm::device_uvector<value_idx> children;
  rmm::device_uvector<value_t> lambdas;
  rmm::device_uvector<value_idx> sizes;

  size_t n_edges;
  size_t n_leaves;
  int n_clusters;
};

};  // namespace Common
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML