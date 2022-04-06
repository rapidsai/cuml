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

#include <cub/cub.cuh>

#include <raft/cudart_utils.h>

#include <raft/sparse/convert/csr.hpp>
#include <raft/sparse/op/sort.hpp>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/label/classlabels.hpp>

#include <algorithm>

#include "../condensed_hierarchy.cu"

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
namespace Utils {

/**
 * Invokes a cub segmented reduce function over a CSR data array
 * using the indptr as segment offsets
 * @tparam value_idx
 * @tparam value_t
 * @tparam CUBReduceFunc
 * @param[in] in data array (size offsets[n_segments]+1)
 * @param[out] out output data array (size offsets[n_segmented]+1)
 * @param[in] n_segments number of segments in offsets array
 * @param[in] offsets array of segment offsets (size n_segments+1)
 * @param[in] stream cuda stream for ordering operations
 * @param[in] cub_reduce_func segmented reduction function
 */
template <typename value_idx, typename value_t, typename CUBReduceFunc>
void cub_segmented_reduce(const value_t* in,
                          value_t* out,
                          int n_segments,
                          const value_idx* offsets,
                          cudaStream_t stream,
                          CUBReduceFunc cub_reduce_func)
{
  rmm::device_uvector<char> d_temp_storage(0, stream);
  size_t temp_storage_bytes = 0;
  cub_reduce_func(
    nullptr, temp_storage_bytes, in, out, n_segments, offsets, offsets + 1, stream, false);
  d_temp_storage.resize(temp_storage_bytes, stream);

  cub_reduce_func(d_temp_storage.data(),
                  temp_storage_bytes,
                  in,
                  out,
                  n_segments,
                  offsets,
                  offsets + 1,
                  stream,
                  false);
}

/**
 * Constructs a cluster tree from a CondensedHierarchy by
 * filtering for only entries with cluster size > 1
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree condensed hierarchy (size n_leaves + n_clusters)
 * @return a new condensed hierarchy with only entries of size > 1
 */
template <typename value_idx, typename value_t>
Common::CondensedHierarchy<value_idx, value_t> make_cluster_tree(
  const raft::handle_t& handle, Common::CondensedHierarchy<value_idx, value_t>& condensed_tree)
{
  auto stream        = handle.get_stream();
  auto thrust_policy = handle.get_thrust_policy();
  auto parents       = condensed_tree.get_parents();
  auto children      = condensed_tree.get_children();
  auto lambdas       = condensed_tree.get_lambdas();
  auto sizes         = condensed_tree.get_sizes();

  value_idx cluster_tree_edges = thrust::transform_reduce(
    thrust_policy,
    sizes,
    sizes + condensed_tree.get_n_edges(),
    [=] __device__(value_idx a) { return a > 1; },
    0,
    thrust::plus<value_idx>());

  // remove leaves from condensed tree
  rmm::device_uvector<value_idx> cluster_parents(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> cluster_children(cluster_tree_edges, stream);
  rmm::device_uvector<value_t> cluster_lambdas(cluster_tree_edges, stream);
  rmm::device_uvector<value_idx> cluster_sizes(cluster_tree_edges, stream);

  auto in = thrust::make_zip_iterator(thrust::make_tuple(parents, children, lambdas, sizes));

  auto out = thrust::make_zip_iterator(thrust::make_tuple(
    cluster_parents.data(), cluster_children.data(), cluster_lambdas.data(), cluster_sizes.data()));

  thrust::copy_if(thrust_policy,
                  in,
                  in + (condensed_tree.get_n_edges()),
                  sizes,
                  out,
                  [=] __device__(value_idx a) { return a > 1; });

  auto n_leaves = condensed_tree.get_n_leaves();
  thrust::transform(thrust_policy,
                    cluster_parents.begin(),
                    cluster_parents.end(),
                    cluster_parents.begin(),
                    [n_leaves] __device__(value_idx a) { return a - n_leaves; });
  thrust::transform(thrust_policy,
                    cluster_children.begin(),
                    cluster_children.end(),
                    cluster_children.begin(),
                    [n_leaves] __device__(value_idx a) { return a - n_leaves; });

  return Common::CondensedHierarchy<value_idx, value_t>(handle,
                                                        condensed_tree.get_n_leaves(),
                                                        cluster_tree_edges,
                                                        condensed_tree.get_n_clusters(),
                                                        std::move(cluster_parents),
                                                        std::move(cluster_children),
                                                        std::move(cluster_lambdas),
                                                        std::move(cluster_sizes));
}

/**
 * Computes a CSR index of sorted parents of condensed tree.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[inout] condensed_tree cluster tree (condensed hierarchy with all nodes of size > 1)
 * @param[in] sorted_parents parents array sorted
 * @param[out] indptr CSR indptr of parents array after sort
 */
template <typename value_idx, typename value_t>
void parent_csr(const raft::handle_t& handle,
                Common::CondensedHierarchy<value_idx, value_t>& condensed_tree,
                value_idx* sorted_parents,
                value_idx* indptr)
{
  auto stream        = handle.get_stream();
  auto thrust_policy = handle.get_thrust_policy();

  auto children   = condensed_tree.get_children();
  auto sizes      = condensed_tree.get_sizes();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_leaves   = condensed_tree.get_n_leaves();
  auto n_clusters = condensed_tree.get_n_clusters();

  // 0-index sorted parents by subtracting n_leaves for offsets and birth/stability indexing
  auto index_op = [n_leaves] __device__(const auto& x) { return x - n_leaves; };
  thrust::transform(
    thrust_policy, sorted_parents, sorted_parents + n_edges, sorted_parents, index_op);

  raft::sparse::convert::sorted_coo_to_csr(sorted_parents, n_edges, indptr, n_clusters + 1, stream);
}

};  // namespace Utils
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML