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

#include <raft/label/classlabels.cuh>

#include <cub/cub.cuh>

#include <raft/cudart_utils.h>
#include <cuml/common/logger.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <raft/sparse/op/sort.h>
#include <raft/sparse/convert/csr.cuh>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <cuml/cluster/hdbscan.hpp>

namespace ML {
namespace HDBSCAN {
namespace Common {

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t>::CondensedHierarchy(
  const raft::handle_t &handle_, size_t n_leaves_)
  : handle(handle_),
    n_leaves(n_leaves_),
    parents(0, handle.get_stream()),
    children(0, handle.get_stream()),
    lambdas(0, handle.get_stream()),
    sizes(0, handle.get_stream()) {}

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t>::CondensedHierarchy(
  const raft::handle_t &handle_, size_t n_leaves_, int n_edges_,
  value_idx *parents_, value_idx *children_, value_t *lambdas_,
  value_idx *sizes_)
  : handle(handle_),
    n_leaves(n_leaves_),
    n_edges(n_edges_),
    parents(0, handle.get_stream()),
    children(0, handle.get_stream()),
    lambdas(0, handle.get_stream()),
    sizes(0, handle.get_stream()) {
  parents.resize(n_edges_, handle.get_stream());
  children.resize(n_edges_, handle.get_stream());
  lambdas.resize(n_edges_, handle.get_stream());
  sizes.resize(n_edges_, handle.get_stream());

  raft::copy(parents.begin(), parents_, n_edges_, handle.get_stream());
  raft::copy(children.begin(), children_, n_edges_, handle.get_stream());
  raft::copy(lambdas.begin(), lambdas_, n_edges_, handle.get_stream());
  raft::copy(sizes.begin(), sizes_, n_edges_, handle.get_stream());

  auto parents_ptr = thrust::device_pointer_cast(parents.data());

  auto parents_min_max =
    thrust::minmax_element(thrust::cuda::par.on(handle.get_stream()),
                           parents_ptr, parents_ptr + n_edges);
  auto min_cluster = *parents_min_max.first;
  auto max_cluster = *parents_min_max.second;

  n_clusters = max_cluster - min_cluster + 1;
}

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t>::CondensedHierarchy(
  const raft::handle_t &handle_, size_t n_leaves_, int n_edges_,
  int n_clusters_, rmm::device_uvector<value_idx> &&parents_,
  rmm::device_uvector<value_idx> &&children_,
  rmm::device_uvector<value_t> &&lambdas_,
  rmm::device_uvector<value_idx> &&sizes_)
  : handle(handle_),
    n_leaves(n_leaves_),
    n_edges(n_edges_),
    n_clusters(n_clusters_),
    parents(std::move(parents_)),
    children(std::move(children_)),
    lambdas(std::move(lambdas_)),
    sizes(std::move(sizes_)) {}

/**
 * Populates the condensed hierarchy object with the output
 * from Condense::condense_hierarchy
 * @param full_parents
 * @param full_children
 * @param full_lambdas
 * @param full_sizes
 */
template <typename value_idx, typename value_t>
void CondensedHierarchy<value_idx, value_t>::condense(value_idx *full_parents,
                                                      value_idx *full_children,
                                                      value_t *full_lambdas,
                                                      value_idx *full_sizes,
                                                      value_idx size) {
  auto stream = handle.get_stream();

  if (size == -1) size = 4 * (n_leaves - 1) + 2;

  n_edges = thrust::transform_reduce(
    thrust::cuda::par.on(stream), full_sizes, full_sizes + size,
    [=] __device__(value_idx a) { return a != -1; }, 0,
    thrust::plus<value_idx>());

  parents.resize(n_edges, stream);
  children.resize(n_edges, stream);
  lambdas.resize(n_edges, stream);
  sizes.resize(n_edges, stream);

  auto in = thrust::make_zip_iterator(
    thrust::make_tuple(full_parents, full_children, full_lambdas, full_sizes));

  auto out = thrust::make_zip_iterator(thrust::make_tuple(
    parents.data(), children.data(), lambdas.data(), sizes.data()));

  thrust::copy_if(
    thrust::cuda::par.on(stream), in, in + size, out,
    [=] __device__(
      thrust::tuple<value_idx, value_idx, value_t, value_idx> tup) {
      return thrust::get<3>(tup) != -1;
    });

  // TODO: Avoid the copies here by updating kernel
  rmm::device_uvector<value_idx> parent_child(n_edges * 2, stream);
  raft::copy_async(parent_child.begin(), children.begin(), n_edges, stream);
  raft::copy_async(parent_child.begin() + n_edges, parents.begin(), n_edges,
                   stream);

  // find n_clusters
  auto parents_ptr = thrust::device_pointer_cast(parents.data());
  auto max_parent = *(thrust::max_element(thrust::cuda::par.on(stream),
                                          parents_ptr, parents_ptr + n_edges));

  // now invert labels
  auto invert_op = [max_parent, n_leaves = n_leaves] __device__(auto &x) {
    return x >= n_leaves ? max_parent - x + n_leaves : x;
  };

  thrust::transform(thrust::cuda::par.on(stream), parent_child.begin(),
                    parent_child.end(), parent_child.begin(), invert_op);

  raft::label::make_monotonic(parent_child.data(), parent_child.data(),
                              parent_child.size(), stream,
                              handle.get_device_allocator(), true);

  raft::copy_async(children.begin(), parent_child.begin(), n_edges, stream);
  raft::copy_async(parents.begin(), parent_child.begin() + n_edges, n_edges,
                   stream);

  auto parents_min_max = thrust::minmax_element(
    thrust::cuda::par.on(stream), parents_ptr, parents_ptr + n_edges);
  auto min_cluster = *parents_min_max.first;
  auto max_cluster = *parents_min_max.second;

  n_clusters = max_cluster - min_cluster + 1;
}

template <typename value_idx, typename value_t>
value_idx CondensedHierarchy<value_idx, value_t>::get_cluster_tree_edges() {
  return thrust::transform_reduce(
    thrust::cuda::par.on(handle.get_stream()), get_sizes(),
    get_sizes() + get_n_edges(), [=] __device__(value_t a) { return a > 1; }, 0,
    thrust::plus<value_idx>());
}

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t> make_cluster_tree(
  const raft::handle_t &handle,
  CondensedHierarchy<value_idx, value_t> &condensed_tree) {
  auto stream = handle.get_stream();
  auto thrust_policy = rmm::exec_policy(stream);
  auto parents = condensed_tree.get_parents();
  auto children = condensed_tree.get_children();
  auto lambdas = condensed_tree.get_lambdas();
  auto sizes = condensed_tree.get_sizes();

  value_idx cluster_tree_edges = thrust::transform_reduce(
    thrust_policy, sizes, sizes + condensed_tree.get_n_edges(),
    [=] __device__(value_idx a) { return a > 1; }, 0,
    thrust::plus<value_idx>());

  // remove leaves from condensed tree
  rmm::device_uvector<value_idx> cluster_parents(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> cluster_children(cluster_tree_edges, stream);
  rmm::device_uvector<value_t> cluster_lambdas(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> cluster_sizes(cluster_tree_edges, stream);

  auto in = thrust::make_zip_iterator(
    thrust::make_tuple(parents, children, lambdas, sizes));

  auto out = thrust::make_zip_iterator(
    thrust::make_tuple(cluster_parents.data(), cluster_children.data(),
                       cluster_lambdas.data(), cluster_sizes.data()));

  thrust::copy_if(thrust_policy, in, in + (condensed_tree.get_n_edges()), sizes,
                  out, [=] __device__(value_idx a) { return a > 1; });

  auto n_leaves = condensed_tree.get_n_leaves();
  thrust::transform(
    thrust_policy, cluster_parents.begin(), cluster_parents.end(),
    cluster_parents.begin(),
    [n_leaves] __device__(value_idx a) { return a - n_leaves; });
  thrust::transform(
    thrust_policy, cluster_children.begin(), cluster_children.end(),
    cluster_children.begin(),
    [n_leaves] __device__(value_idx a) { return a - n_leaves; });

  return CondensedHierarchy<value_idx, value_t>(
    handle, condensed_tree.get_n_leaves(), cluster_tree_edges,
    condensed_tree.get_n_clusters(), std::move(cluster_parents),
    std::move(cluster_children), std::move(cluster_lambdas),
    std::move(cluster_sizes));
}

};  // namespace Common
};  // namespace HDBSCAN
};  // namespace ML
