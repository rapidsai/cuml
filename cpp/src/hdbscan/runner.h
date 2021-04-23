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

#include <raft/cudart_utils.h>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/sparse/hierarchy/detail/agglomerative.cuh>
#include <raft/sparse/hierarchy/detail/mst.cuh>

#include "reachability.cuh"
#include "tree.cuh"

namespace ML {
namespace HDBSCAN {

template <typename value_idx, typename value_t>
struct MSTEpilogueReachability {
  MSTEpilogueReachability(value_idx m_, value_t *core_distances_)
    : core_distances(core_distances_), m(m_) {}

  void operator()(const raft::handle_t &handle, value_idx *coo_rows,
                  value_idx *coo_cols, value_t *coo_data, value_idx nnz) {
    auto first = thrust::make_zip_iterator(
      thrust::make_tuple(coo_rows, coo_cols, coo_data));
    thrust::transform(
      thrust::cuda::par.on(handle.get_stream()), first, first + nnz, coo_data,
      [=] __device__(thrust::tuple<value_idx, value_idx, value_t> t) {
        return max(core_distances[thrust::get<0>(t)],
                   core_distances[thrust::get<1>(t)], thrust::get<2>(t));
      });
  }

 private:
  value_t *core_distances;
  value_idx m;
};

template <typename value_idx = int64_t, typename value_t = float>
void _fit(const raft::handle_t &handle, value_t *X, value_idx m, value_idx n,
          raft::distance::DistanceType metric, int k, int min_pts, float alpha,
          int min_cluster_size) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  /**
   * Mutual reachability graph
   */
  rmm::device_uvector<value_idx> mutual_reachability_graph_inds(k * m, stream);
  rmm::device_uvector<value_t> mutual_reachability_graph_dists(k * m, stream);
  rmm::device_uvector<value_t> core_dists(k * m, stream);

  Reachability::mutual_reachability_dists(
    handle, X, m, n, metric, min_pts, k, mutual_reachability_graph_inds.data(),
    mutual_reachability_graph_dists.data(), core_dists.data());

  /**
   * Construct MST sorted by weights
   */
  rmm::device_uvector<value_idx> mst_rows(m - 1, stream);
  rmm::device_uvector<value_t> mst_cols(m - 1, stream);
  rmm::device_uvector<value_idx> mst_data(m - 1, stream);

  // during knn graph connection
  raft::hierarchy::detail::build_sorted_mst(
    handle, X, mutual_reachability_graph_inds.data(),
    mutual_reachability_graph_dists.data(), m, n, mst_rows, mst_cols, mst_data,
    k * m, metric, 10, MSTEpilogueReachability<value_idx, value_t>());

  /**
   * Perform hierarchical labeling
   */
  value_idx n_edges = m - 1;

  rmm::device_uvector<value_idx> out_src(n_edges, stream);
  rmm::device_uvector<value_idx> out_dst(n_edges, stream);
  rmm::device_uvector<value_t> out_delta(n_edges, stream);
  rmm::device_uvector<value_idx> out_size(n_edges, stream);

  raft::hierarchy::detail::build_dendrogram_host(
    mst_rows.data(), mst_cols.data(), mst_data.data(), n_edges, out_src.data(),
    out_dst.data(), out_delta.data(), out_size.data());

  /**
   * Condense branches of tree according to min cluster size
   */
  Tree::CondensedHierarchy<value_idx, value_t> condensed_tree(m, stream);
  condense_hierarchy(handle, out_src.data(), out_dst.data(), out_delta.data(),
                     out_size.data(), min_cluster_size, m, condensed_tree);

  rmm::device_uvector<value_t> stabilities(condensed_tree.get_n_clusters(),
                                           handle.get_stream());
  compute_stabilities(handle, condensed_tree, stabilities);

  /**
   * Extract labels from stability
   */
}

};  // end namespace HDBSCAN
};  // end namespace ML