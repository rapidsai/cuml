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

#include "hierarchy/distance.cuh"
#include "hierarchy/agglomerative.h"
#include "hierarchy/mst.cuh"

namespace ML {
namespace Linkage {


template <typename value_idx, typename value_t>
void get_distance_graph(const raft::handle_t &handle,
                        const value_t *X, size_t m, size_t n,
                        raft::distance::DistanceType metric,
                        LinkageDistance dist_type,
                        raft::mr::device::buffer<value_idx> &indptr,
                        raft::mr::device::buffer<value_idx> &indices,
                        raft::mr::device::buffer<value_t> &data,
                        int c) {

  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  indptr.resize(m+1, stream);

  switch(dist_type) {

    case LinkageDistance::PAIRWISE:

      indices.resize(m*m, stream);
      data.resize(m*m, stream);

      Distance::pairwise_distances(handle, X, m, n, metric, indptr.data(),
                                   indices.data(), data.data());
      break;

    case LinkageDistance::KNN_GRAPH:

    {
      int k = Distance::build_k(m, c);

      // knn graph is symmetric
      MLCommon::Sparse::COO<value_t, value_idx> knn_graph_coo(d_alloc, stream);

      Distance::knn_graph(handle, X, m, n, metric, knn_graph_coo, c);

      MLCommon::Sparse::coo_sort(&knn_graph_coo, d_alloc, stream);

      indices.resize(knn_graph_coo.nnz, stream);
      data.resize(knn_graph_coo.nnz, stream);

      MLCommon::Sparse::sorted_coo_to_csr(&knn_graph_coo, indptr.data(),
                                          d_alloc, stream);

      raft::copy_async(indices.data(), knn_graph_coo.cols(), knn_graph_coo.nnz, stream);
      raft::copy_async(data.data(), knn_graph_coo.vals(), knn_graph_coo.nnz, stream);
    }

      break;

    default:
      throw raft::exception("Unsupported linkage distance");
  }

}


template <typename value_idx, typename value_t>
void _single_linkage(const raft::handle_t &handle,
                     const value_t *X,
                     size_t m,
                     size_t n,
                     raft::distance::DistanceType metric,
                     LinkageDistance dist_type,
                     linkage_output<value_idx, value_t> *out,
                     int c) {

  auto stream = handle.get_stream();
  auto d_alloc = handle.get_device_allocator();

  raft::print_device_vector("X: ", X, 5, std::cout);


  CUML_LOG_INFO("Running distances");

  raft::mr::device::buffer<value_idx> indptr(d_alloc, stream, 0);
  raft::mr::device::buffer<value_idx> indices(d_alloc, stream, 0);
  raft::mr::device::buffer<value_t> pw_dists(d_alloc, stream, 0);

  /**
   * 1. Construct distance graph
   */
  get_distance_graph(handle, X, m, n, metric, dist_type, indptr, indices,
                     pw_dists, c);

  raft::mr::device::buffer<value_idx> mst_rows(d_alloc, stream, 0);
  raft::mr::device::buffer<value_idx> mst_cols(d_alloc, stream, 0);
  raft::mr::device::buffer<value_t> mst_data(d_alloc, stream, 0);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  raft::print_device_vector("indptr: ", indptr.data(), 50, std::cout);
  raft::print_device_vector("indices: ", indices.data(), 50, std::cout);
  raft::print_device_vector("data: ", pw_dists.data(), 50, std::cout);

  CUML_LOG_INFO("Constructing MST");

  /**
   * 2. Construct MST, sorted by weights
   */
  MST::build_sorted_mst<value_idx, value_t>(handle,
                                            indptr.data(),
                                            indices.data(),
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