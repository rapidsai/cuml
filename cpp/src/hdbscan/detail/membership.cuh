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

#include "kernels/membership.cuh"
#include "utils.h"

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

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Membership {

// TODO: Compute outlier scores

template <typename value_idx, typename value_t>
void get_probabilities(
  const raft::handle_t &handle,
  Common::CondensedHierarchy<value_idx, value_t> &condensed_tree,
  const value_idx *labels, value_t *probabilities) {
  auto stream = handle.get_stream();
  auto exec_policy = rmm::exec_policy(stream);

  auto parents = condensed_tree.get_parents();
  auto children = condensed_tree.get_children();
  auto lambdas = condensed_tree.get_lambdas();
  auto n_edges = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves = condensed_tree.get_n_leaves();

  rmm::device_uvector<value_idx> sorted_parents(n_edges, stream);
  raft::copy_async(sorted_parents.data(), parents, n_edges, stream);

  // 0-index sorted parents by subtracting n_leaves for offsets and birth/stability indexing
  auto index_op = [n_leaves] __device__(const auto &x) { return x - n_leaves; };
  thrust::transform(exec_policy, sorted_parents.begin(), sorted_parents.end(),
                    sorted_parents.begin(), index_op);

  rmm::device_uvector<value_idx> sorted_parents_offsets(n_edges + 1, stream);
  raft::sparse::convert::sorted_coo_to_csr(
    sorted_parents.data(), n_edges, sorted_parents_offsets.data(),
    n_clusters + 1, handle.get_device_allocator(), handle.get_stream());

  // this is to find maximum lambdas of all children under a prent
  rmm::device_uvector<value_t> deaths(n_clusters, stream);
  thrust::fill(exec_policy, deaths.begin(), deaths.end(), 0.0f);

  Utils::cub_segmented_reduce(
    lambdas, deaths.data(), n_clusters, sorted_parents_offsets.data(), stream,
    cub::DeviceSegmentedReduce::Max<const value_t *, value_t *,
                                    const value_idx *>);

  raft::print_device_vector("deaths", deaths.data(), deaths.size(), std::cout);
  // Calculate probability per point
  thrust::fill(exec_policy, probabilities, probabilities + n_leaves, 0.0f);

  probabilities_functor<value_idx, value_t> probabilities_op(
    probabilities, deaths.data(), children, lambdas, labels, n_leaves);
  thrust::for_each(exec_policy, thrust::make_counting_iterator(value_idx(0)),
                   thrust::make_counting_iterator(n_edges), probabilities_op);
}

};  // namespace Membership
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML