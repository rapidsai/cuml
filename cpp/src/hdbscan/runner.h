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

#include <cuml/cuml_api.h>
#include <raft/cudart_utils.h>
#include <common/cumlHandle.hpp>

#include <raft/mr/device/buffer.hpp>

#include <hierarchy/agglomerative.h>
#include "hierarchy/mst.cuh"
#include "reachability.cuh"

namespace ML {
namespace HDBSCAN {

template <typename value_idx = int64_t, typename value_t = float>
void _fit(const raft::handle_t &handle,
          value_t *X,
          value_idx m,
          value_idx n,
          raft::distance::DistanceType metric,
          int k,
          int min_pts,
          float alpha) {

  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  /**
   * Mutual reachability graph
   */
  raft::mr::device::buffer<value_idx> pw_dists(d_alloc, stream, m*m);

  // @TODO: reduce this memory cost in the future iterations
  Reachability::pairwise_mutual_reachability_graph(handle,
                                                   X,
                                                   m,
                                                   n,
                                                   metric,
                                                   min_pts,
                                                   k,
                                                   min_pts,
                                                   pw_dists.data());

  /**
   * Construct MST sorted by weights
   */
  raft::mr::device::buffer<value_idx> mst_rows(d_alloc, stream, m-1);
  raft::mr::device::buffer<value_t> mst_cols(d_alloc, stream, m-1);
  raft::mr::device::buffer<value_idx> mst_data(d_alloc, stream, m-1);

  Linkage::MST::build_sorted_mst(handle,
                        pw_dists.data(),
                        m,
                        mst_rows.data(),
                        mst_cols.data(),
                        mst_data.data());

  /**
   * Perform hierarchical labeling
   */
  value_idx n_edges = m-1;

  raft::mr::device::buffer<value_idx> out_src(d_alloc, stream, n_edges);
  raft::mr::device::buffer<value_idx> out_dst(d_alloc, stream, n_edges);
  raft::mr::device::buffer<value_t> out_delta(d_alloc, stream, n_edges);
  raft::mr::device::buffer<value_idx> out_size(d_alloc, stream, n_edges);

  Linkage::Label::Agglomerative::label_hierarchy_host(mst_rows.data(),
                                                      mst_cols.data(),
                                                      mst_data.data(),
                                                      n_edges,
                                                      out_src.data(),
                                                      out_dst.data(),
                                                      out_delta.data(),
                                                      out_size.data());


  /**
   * Condense branches of tree according to min cluster size
   */



  /**
   * Extract labels from stability
   */

}

}; // end namespace HDBSCAN
}; // end namespace ML