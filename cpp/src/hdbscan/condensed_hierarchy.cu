/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/logger.hpp>

#include <raft/label/classlabels.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace ML {
namespace HDBSCAN {
namespace Common {

struct TupleComp {
  template <typename one, typename two>
  __host__ __device__ bool operator()(const one& t1, const two& t2)
  {
    // sort first by each parent,
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;

    // within each parent, sort by each child,
    if (thrust::get<1>(t1) < thrust::get<1>(t2)) return true;
    if (thrust::get<1>(t1) > thrust::get<1>(t2)) return false;

    // then sort by value in descending order
    return thrust::get<2>(t1) < thrust::get<2>(t2);
  }
};

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t>::CondensedHierarchy(const raft::handle_t& handle_,
                                                           size_t n_leaves_)
  : handle(handle_),
    n_leaves(n_leaves_),
    parents(0, handle.get_stream()),
    children(0, handle.get_stream()),
    lambdas(0, handle.get_stream()),
    sizes(0, handle.get_stream())
{
}

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t>::CondensedHierarchy(const raft::handle_t& handle_,
                                                           size_t n_leaves_,
                                                           int n_edges_,
                                                           value_idx* parents_,
                                                           value_idx* children_,
                                                           value_t* lambdas_,
                                                           value_idx* sizes_)
  : handle(handle_),
    n_leaves(n_leaves_),
    n_edges(n_edges_),
    parents(n_edges_, handle.get_stream()),
    children(n_edges_, handle.get_stream()),
    lambdas(n_edges_, handle.get_stream()),
    sizes(n_edges_, handle.get_stream())
{
  raft::copy(parents.begin(), parents_, n_edges_, handle.get_stream());
  raft::copy(children.begin(), children_, n_edges_, handle.get_stream());
  raft::copy(lambdas.begin(), lambdas_, n_edges_, handle.get_stream());
  raft::copy(sizes.begin(), sizes_, n_edges_, handle.get_stream());

  auto parents_ptr = thrust::device_pointer_cast(parents.data());

  auto parents_min_max = thrust::minmax_element(
    thrust::cuda::par.on(handle.get_stream()), parents_ptr, parents_ptr + n_edges);
  auto min_cluster = *parents_min_max.first;
  auto max_cluster = *parents_min_max.second;

  n_clusters = max_cluster - min_cluster + 1;

  auto sort_keys =
    thrust::make_zip_iterator(thrust::make_tuple(parents.begin(), children.begin(), sizes.begin()));
  auto sort_values = thrust::make_zip_iterator(thrust::make_tuple(lambdas.begin()));

  thrust::sort_by_key(thrust::cuda::par.on(handle.get_stream()),
                      sort_keys,
                      sort_keys + n_edges,
                      sort_values,
                      TupleComp());
}

template <typename value_idx, typename value_t>
CondensedHierarchy<value_idx, value_t>::CondensedHierarchy(
  const raft::handle_t& handle_,
  size_t n_leaves_,
  int n_edges_,
  int n_clusters_,
  rmm::device_uvector<value_idx>&& parents_,
  rmm::device_uvector<value_idx>&& children_,
  rmm::device_uvector<value_t>&& lambdas_,
  rmm::device_uvector<value_idx>&& sizes_)
  : handle(handle_),
    n_leaves(n_leaves_),
    n_edges(n_edges_),
    n_clusters(n_clusters_),
    parents(std::move(parents_)),
    children(std::move(children_)),
    lambdas(std::move(lambdas_)),
    sizes(std::move(sizes_))
{
}

/**
 * Populates the condensed hierarchy object with the output
 * from Condense::condense_hierarchy
 * @param full_parents
 * @param full_children
 * @param full_lambdas
 * @param full_sizes
 */
template <typename value_idx, typename value_t>
void CondensedHierarchy<value_idx, value_t>::condense(value_idx* full_parents,
                                                      value_idx* full_children,
                                                      value_t* full_lambdas,
                                                      value_idx* full_sizes,
                                                      value_idx size)
{
  auto stream = handle.get_stream();

  if (size == -1) size = 4 * (n_leaves - 1) + 2;

  n_edges = thrust::transform_reduce(
    thrust::cuda::par.on(stream),
    full_sizes,
    full_sizes + size,
    cuda::proclaim_return_type<value_idx>(
      [=] __device__(value_idx a) -> value_idx { return static_cast<value_idx>(a != -1); }),
    static_cast<value_idx>(0),
    cuda::std::plus<value_idx>());

  parents.resize(n_edges, stream);
  children.resize(n_edges, stream);
  lambdas.resize(n_edges, stream);
  sizes.resize(n_edges, stream);

  auto in = thrust::make_zip_iterator(
    thrust::make_tuple(full_parents, full_children, full_lambdas, full_sizes));

  auto out = thrust::make_zip_iterator(
    thrust::make_tuple(parents.data(), children.data(), lambdas.data(), sizes.data()));

  thrust::copy_if(thrust::cuda::par.on(stream),
                  in,
                  in + size,
                  out,
                  [=] __device__(thrust::tuple<value_idx, value_idx, value_t, value_idx> tup) {
                    return thrust::get<3>(tup) != -1;
                  });

  // TODO: Avoid the copies here by updating kernel
  rmm::device_uvector<value_idx> parent_child(n_edges * 2, stream);
  raft::copy_async(parent_child.begin(), children.begin(), n_edges, stream);
  raft::copy_async(parent_child.begin() + n_edges, parents.begin(), n_edges, stream);

  // find n_clusters
  auto parents_ptr = thrust::device_pointer_cast(parents.data());
  auto max_parent =
    *(thrust::max_element(thrust::cuda::par.on(stream), parents_ptr, parents_ptr + n_edges));

  // now invert labels
  auto invert_op = [max_parent, n_leaves = n_leaves] __device__(auto& x) {
    return x >= n_leaves ? max_parent - x + n_leaves : x;
  };

  thrust::transform(thrust::cuda::par.on(stream),
                    parent_child.begin(),
                    parent_child.end(),
                    parent_child.begin(),
                    invert_op);

  raft::label::make_monotonic(
    parent_child.data(), parent_child.data(), parent_child.size(), stream, true);

  raft::copy_async(children.begin(), parent_child.begin(), n_edges, stream);
  raft::copy_async(parents.begin(), parent_child.begin() + n_edges, n_edges, stream);

  auto parents_min_max =
    thrust::minmax_element(thrust::cuda::par.on(stream), parents_ptr, parents_ptr + n_edges);
  auto min_cluster = *parents_min_max.first;
  auto max_cluster = *parents_min_max.second;

  n_clusters = max_cluster - min_cluster + 1;

  auto sort_keys =
    thrust::make_zip_iterator(thrust::make_tuple(parents.begin(), children.begin(), sizes.begin()));
  auto sort_values = thrust::make_zip_iterator(thrust::make_tuple(lambdas.begin()));

  thrust::sort_by_key(
    thrust::cuda::par.on(stream), sort_keys, sort_keys + n_edges, sort_values, TupleComp());
}

};  // namespace Common
};  // namespace HDBSCAN
};  // namespace ML
