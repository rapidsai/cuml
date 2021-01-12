/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>

#include <cuml/cuml_api.h>
#include <common/cumlHandle.hpp>

#include <raft/mr/device/buffer.hpp>

#include <cuml/cluster/linkage.hpp>

#include <distance/distance.cuh>
#include <sparse/coo.cuh>

#include <hierarchy/agglomerative.cuh>
#include <hierarchy/connectivities.cuh>
#include <hierarchy/mst.cuh>

namespace ML {
namespace Linkage {

template <typename value_idx, typename value_t>
void _single_linkage(const raft::handle_t &handle, const value_t *X, size_t m,
                     size_t n, raft::distance::DistanceType metric,
                     LinkageDistance dist_type,
                     linkage_output<value_idx, value_t> *out, int c,
                     int n_clusters) {
  auto stream = handle.get_stream();
  auto d_alloc = handle.get_device_allocator();

  raft::mr::device::buffer<value_idx> indptr(d_alloc, stream, 0);
  raft::mr::device::buffer<value_idx> indices(d_alloc, stream, 0);
  raft::mr::device::buffer<value_t> pw_dists(d_alloc, stream, 0);

  /**
   * 1. Construct distance graph
   */
  Distance::get_distance_graph(handle, X, m, n, metric, dist_type, indptr,
                               indices, pw_dists, c);

  raft::mr::device::buffer<value_idx> mst_rows(d_alloc, stream, 0);
  raft::mr::device::buffer<value_idx> mst_cols(d_alloc, stream, 0);
  raft::mr::device::buffer<value_t> mst_data(d_alloc, stream, 0);
  /**
   * 2. Construct MST, sorted by weights
   */
  MST::build_sorted_mst<value_idx, value_t>(handle, indptr.data(),
                                            indices.data(), pw_dists.data(), m,
                                            mst_rows, mst_cols, mst_data);

  pw_dists.release();

  /**
   * Perform hierarchical labeling
   */
  size_t n_edges = mst_rows.size();

  raft::mr::device::buffer<value_idx> children(d_alloc, stream, n_edges * 2);
  raft::mr::device::buffer<value_t> out_delta(d_alloc, stream, n_edges);
  raft::mr::device::buffer<value_idx> out_size(d_alloc, stream, n_edges);

  // Create dendrogram
  Label::Agglomerative::build_dendrogram_host<value_idx, value_t>(
    handle, mst_rows.data(), mst_cols.data(), mst_data.data(), n_edges,
    children, out_delta, out_size);

  Label::Agglomerative::extract_flattened_clusters(handle, out->labels,
                                                   children, n_clusters, m);
}

};  // end namespace Linkage
};  // end namespace ML