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

#include "distance/distance.cuh"
#include "hierarchy/agglomerative.h"
#include "hierarchy/mst.cuh"

namespace ML {
namespace Linkage {

template <typename value_idx, typename value_t>
void _single_linkage(const raft::handle_t &handle,
                     const value_t *X,
                     size_t m,
                     size_t n,
                     raft::distance::DistanceType metric,
                     linkage_output<value_idx, value_t> *out) {

  auto stream = handle.get_stream();
  auto d_alloc = handle.get_device_allocator();

  raft::print_device_vector("X: ", X, m*n, std::cout);


  CUML_LOG_INFO("Running pairwise distances");

  raft::mr::device::buffer<value_t> pw_dists(d_alloc, stream, m*m);
  raft::mr::device::buffer<char> workspace(d_alloc, stream, 0);
  raft::mr::device::buffer<value_idx> mst_rows(d_alloc, stream, m-1);
  raft::mr::device::buffer<value_idx> mst_cols(d_alloc, stream, m-1);
  raft::mr::device::buffer<value_t> mst_data(d_alloc, stream, m-1);

  /**
   * Construct pairwise distances
   */

  // @TODO: This is super expensive. Future versions need to eliminate
  //   the pairwise distance matrix, use KNN, or an MST based on the KNN graph
  MLCommon::Distance::pairwise_distance<value_t, size_t>(X, X, pw_dists.data(),
                                                         m, m, n, workspace,
                                                         metric, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  raft::print_device_vector("data: ", pw_dists.data(), 2, std::cout);


  CUML_LOG_INFO("Constructing MST");
  /**
   * Construct MST sorted by weights
   */

  MST::build_sorted_mst<value_idx, value_t>(handle,
                                            pw_dists.data(),
                                            m,
                                            mst_rows.data(),
                                            mst_cols.data(),
                                            mst_data.data());
  pw_dists.release();


  CUML_LOG_INFO("Perform labeling");

  /**
   * Perform hierarchical labeling
   */
  size_t n_edges = m-1;

  raft::mr::device::buffer<value_t> out_delta(d_alloc, stream, n_edges);
  raft::mr::device::buffer<value_idx> out_size(d_alloc, stream, n_edges);

  // @TODO: Label in parallel on device
  Linkage::Label::Agglomerative::label_hierarchy_host
    <value_idx, value_t>(handle,
                         mst_rows.data(),
                         mst_cols.data(),
                         mst_data.data(),
                         n_edges,
                         out->children,
                         out_delta.data(),
                         out_size.data());
}

}; // end namespace Linkage
}; // end namespace ML