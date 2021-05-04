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

#include <hdbscan/condensed_hierarchy.cu>

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

template <typename value_idx, typename value_t, typename CUBReduceFunc>
void segmented_reduce(const value_t *in, value_t *out, int n_segments,
                      const value_idx *offsets, cudaStream_t stream,
                      CUBReduceFunc cub_reduce_func) {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub_reduce_func(d_temp_storage, temp_storage_bytes, in, out, n_segments,
                  offsets, offsets + 1, stream, false);
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub_reduce_func(d_temp_storage, temp_storage_bytes, in, out, n_segments,
                  offsets, offsets + 1, stream, false);
  CUDA_CHECK(cudaFree(d_temp_storage));
}

template <typename value_idx, typename value_t>
Common::CondensedHierarchy<value_idx, value_t> make_cluster_tree(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree) {
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

  return Common::CondensedHierarchy<value_idx, value_t>(
    handle, condensed_tree.get_n_leaves(), cluster_tree_edges,
    condensed_tree.get_n_clusters(), std::move(cluster_parents),
    std::move(cluster_children), std::move(cluster_lambdas),
    std::move(cluster_sizes));
}

};
};
};
};