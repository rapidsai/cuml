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

#include <raft/label/classlabels.cuh>

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
      sizes(0, handle.get_stream()){}

  /**
   * Populates the condensed hierarchy object with the output
   * from Condense::condense_hierarchy
   * @param full_parents
   * @param full_children
   * @param full_lambdas
   * @param full_sizes
   */
  void condense(value_idx *full_parents, value_idx *full_children,
                value_t *full_lambdas, value_idx *full_sizes, value_idx size = -1) {
    auto stream = handle.get_stream();

    if(size == -1)
      size = 4 * (n_leaves - 1) + 2;

    CUML_LOG_DEBUG("calling transform_reduce");
    n_edges = thrust::transform_reduce(
      thrust::cuda::par.on(stream), full_sizes,
      full_sizes + size,
      [=] __device__(value_idx a) { return a != -1; }, 0,
      thrust::plus<value_idx>());

    CUML_LOG_DEBUG("resizing parents");
    parents.resize(n_edges, stream);
    children.resize(n_edges, stream);
    lambdas.resize(n_edges, stream);
    sizes.resize(n_edges, stream);


    auto in = thrust::make_zip_iterator(thrust::make_tuple(
      full_parents, full_children, full_lambdas, full_sizes));

    auto out = thrust::make_zip_iterator(thrust::make_tuple(
      parents.data(), children.data(), lambdas.data(), sizes.data()));

    CUML_LOG_DEBUG("Calling copy_if");
    thrust::copy_if(
      thrust::cuda::par.on(stream), in, in + size, out,
      [=] __device__(
        thrust::tuple<value_idx, value_idx, value_t, value_idx> tup) {
        return thrust::get<3>(tup) != -1;
      });

    raft::print_device_vector("Parents before monotonic", parents.data(), n_edges, std::cout);
    raft::print_device_vector("Children before monotonic", children.data(), n_edges, std::cout);

    // TODO: Avoid the copies here by updating kernel
    rmm::device_uvector<value_idx> parent_child(n_edges * 2, stream);
    raft::copy_async(parent_child.begin(), children.begin(), n_edges, stream);
    raft::copy_async(parent_child.begin() + n_edges, parents.begin(), n_edges, stream);

      // find n_clusters
    auto parents_ptr = thrust::device_pointer_cast(parents.data());
    auto parents_min_max = thrust::minmax_element(thrust::cuda::par.on(stream), parents_ptr, parents_ptr + n_edges);
    auto min_parent = *parents_min_max.first;
    auto max_parent = *parents_min_max.second;

    n_clusters = max_parent - min_parent + 1;

    // now invert labels
    auto invert_op = [max_parent, n_leaves = n_leaves] __device__ (auto &x) {
      return x >= n_leaves ? max_parent - x + n_leaves : x;
    };

    thrust::transform(thrust::cuda::par.on(stream), parent_child.begin(), parent_child.end(), parent_child.begin(), invert_op);

    raft::label::make_monotonic(
         parent_child.data(), parent_child.data(), parent_child.size(), stream, handle.get_device_allocator(), true);
  
    raft::copy_async(children.begin(), parent_child.begin(), n_edges, stream);
    raft::copy_async(parents.begin(), parent_child.begin() + n_edges, n_edges, stream);

    raft::print_device_vector("Parents After transform and monotonic", parent_child.data() + n_edges, n_edges, std::cout);
    raft::print_device_vector("Children After transform monotonic", parent_child.data(), n_edges, std::cout);
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
      get_sizes() + get_n_edges(), [=] __device__(value_t a) { return a > 1; },
      0, thrust::plus<value_idx>());
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
