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

#pragma once

#include "kernels/condense.cuh"

#include <cuml/cluster/hdbscan.hpp>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Condense {

/**
 * Condenses a binary single-linkage tree dendrogram in the Scipy hierarchy
 * format by collapsing subtrees that fall below a minimum cluster size.
 *
 * For increased parallelism, the output array sizes are held fixed but
 * the result will be sparse (e.g. zeros in place of parents who have been
 * removed / collapsed). This function accepts an empty instance of
 * `CondensedHierarchy` and invokes the `condense()` function on it to
 * convert the sparse output arrays into their dense form.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param handle
 * @param[in] children parents/children from single-linkage dendrogram
 * @param[in] delta distances from single-linkage dendrogram
 * @param[in] sizes sizes from single-linkage dendrogram
 * @param[in] min_cluster_size any subtrees less than this size will be
 *                             collapsed.
 * @param[in] n_leaves number of actual data samples in the dendrogram
 * @param[out] condensed_tree output dendrogram. will likely no longer be
 *                            a binary tree.
 */
template <typename value_idx, typename value_t, int tpb = 256>
void build_condensed_hierarchy(const raft::handle_t& handle,
                               const value_idx* children,
                               const value_t* delta,
                               const value_idx* sizes,
                               int min_cluster_size,
                               int n_leaves,
                               Common::CondensedHierarchy<value_idx, value_t>& condensed_tree)
{
  cudaStream_t stream = handle.get_stream();
  auto exec_policy    = handle.get_thrust_policy();

  // Root is the last edge in the dendrogram
  value_idx root = 2 * (n_leaves - 1);

  auto d_ptr           = thrust::device_pointer_cast(children);
  value_idx n_vertices = *(thrust::max_element(exec_policy, d_ptr, d_ptr + root)) + 1;

  // Prevent potential infinite loop from labeling disconnected
  // connectivities graph.
  RAFT_EXPECTS(n_vertices == root,
               "Multiple components found in MST or MST is invalid. "
               "Cannot find single-linkage solution. Found %d vertices "
               "total.",
               static_cast<int>(n_vertices));

  rmm::device_uvector<bool> frontier(root + 1, stream);
  rmm::device_uvector<bool> next_frontier(root + 1, stream);

  thrust::fill(exec_policy, frontier.begin(), frontier.end(), false);
  thrust::fill(exec_policy, next_frontier.begin(), next_frontier.end(), false);

  // Array to propagate the lambda of subtrees actively being collapsed
  // through multiple bfs iterations.
  rmm::device_uvector<value_t> ignore(root + 1, stream);

  // Propagate labels from root
  rmm::device_uvector<value_idx> relabel(root + 1, handle.get_stream());
  thrust::fill(exec_policy, relabel.begin(), relabel.end(), -1);

  raft::update_device(relabel.data() + root, &root, 1, handle.get_stream());

  // Flip frontier for root
  constexpr bool start = true;
  raft::update_device(frontier.data() + root, &start, 1, handle.get_stream());

  rmm::device_uvector<value_idx> out_parent((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_child((root + 1) * 2, stream);
  rmm::device_uvector<value_t> out_lambda((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_size((root + 1) * 2, stream);

  thrust::fill(exec_policy, out_parent.begin(), out_parent.end(), -1);
  thrust::fill(exec_policy, out_child.begin(), out_child.end(), -1);
  thrust::fill(exec_policy, out_lambda.begin(), out_lambda.end(), -1);
  thrust::fill(exec_policy, out_size.begin(), out_size.end(), -1);
  thrust::fill(exec_policy, ignore.begin(), ignore.end(), -1);

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(root + 1, static_cast<value_idx>(tpb));

  value_idx n_elements_to_traverse =
    thrust::reduce(exec_policy, frontier.data(), frontier.data() + root + 1, 0);

  while (n_elements_to_traverse > 0) {
    // TODO: Investigate whether it would be worth performing a gather/argmatch in order
    // to schedule only the number of threads needed. (it might not be worth it)
    condense_hierarchy_kernel<<<grid, tpb, 0, handle.get_stream()>>>(frontier.data(),
                                                                     next_frontier.data(),
                                                                     ignore.data(),
                                                                     relabel.data(),
                                                                     children,
                                                                     delta,
                                                                     sizes,
                                                                     n_leaves,
                                                                     min_cluster_size,
                                                                     out_parent.data(),
                                                                     out_child.data(),
                                                                     out_lambda.data(),
                                                                     out_size.data());

    thrust::copy(exec_policy, next_frontier.begin(), next_frontier.end(), frontier.begin());
    thrust::fill(exec_policy, next_frontier.begin(), next_frontier.end(), false);

    n_elements_to_traverse = thrust::reduce(
      exec_policy, frontier.data(), frontier.data() + root + 1, static_cast<value_idx>(0));

    handle.sync_stream(stream);
  }

  condensed_tree.condense(out_parent.data(), out_child.data(), out_lambda.data(), out_size.data());
}

};  // end namespace Condense
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
