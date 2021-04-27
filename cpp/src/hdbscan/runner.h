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

#include <cuml/common/logger.hpp>

#include <raft/sparse/coo.cuh>
#include <raft/sparse/hierarchy/detail/agglomerative.cuh>
#include <raft/sparse/hierarchy/detail/mst.cuh>

#include "detail/common.h"
#include "detail/condense.cuh"
#include "detail/extract.cuh"
#include "detail/reachability.cuh"

namespace ML {
namespace HDBSCAN {

template <typename value_idx, typename value_t>
struct MSTEpilogueReachability {
  MSTEpilogueReachability(value_idx m_, value_t *core_distances_)
    : core_distances(core_distances_), m(m_) {}

  void operator()(const raft::handle_t &handle, value_idx *coo_rows,
                  value_idx *coo_cols, value_t *coo_data, value_idx nnz) {
    printf("nnz=%d\n", nnz);

    raft::print_device_vector("coo_rows", coo_rows, 2, std::cout);
    raft::print_device_vector("coo_cols", coo_cols, 2, std::cout);
    raft::print_device_vector("coo_data", coo_data, 2, std::cout);
    raft::print_device_vector("core", core_distances, 2, std::cout);

    auto first = thrust::make_zip_iterator(
      thrust::make_tuple(coo_rows, coo_cols, coo_data));
    thrust::transform(
      thrust::cuda::par.on(handle.get_stream()), first, first + nnz, coo_data,
      [=] __device__(thrust::tuple<value_idx, value_idx, value_t> t) {
        return max(core_distances[thrust::get<0>(t)],
                   max(core_distances[thrust::get<1>(t)], thrust::get<2>(t)));
      });

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    CUML_LOG_DEBUG("Executed graph connection");
  }

 private:
  value_t *core_distances;
  value_idx m;
};

template <typename value_idx = int64_t, typename value_t = float>
void _fit(const raft::handle_t &handle, const value_t *X, size_t m, size_t n,
          raft::distance::DistanceType metric, int k, int min_pts,
          int min_cluster_size, hdbscan_output<value_idx, value_t> *out) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  printf("K=%d\n", k);
  printf("min_pts: %d\n", min_pts);

  /**
    * Mutual reachability graph
    */

  rmm::device_uvector<value_idx> mutual_reachability_indptr(m + 1, stream);
  raft::sparse::COO<value_t, value_idx> mutual_reachability_coo(d_alloc, stream,
                                                                k * m * 2);
  rmm::device_uvector<value_t> core_dists(m, stream);

  detail::Reachability::mutual_reachability_graph(
    handle, X, (size_t)m, (size_t)n, metric, k,
    mutual_reachability_indptr.data(), core_dists.data(),
    mutual_reachability_coo);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUML_LOG_DEBUG("Executed mutual reachability");

  /**
    * Construct MST sorted by weights
    */
  rmm::device_uvector<value_idx> mst_rows(m - 1, stream);
  rmm::device_uvector<value_idx> mst_cols(m - 1, stream);
  rmm::device_uvector<value_t> mst_data(m - 1, stream);

  // during knn graph connection
  raft::hierarchy::detail::build_sorted_mst(
    handle, X, mutual_reachability_indptr.data(),
    mutual_reachability_coo.cols(), mutual_reachability_coo.vals(), m, n,
    mst_rows, mst_cols, mst_data, mutual_reachability_coo.nnz,
    MSTEpilogueReachability<value_idx, value_t>(m, core_dists.data()), metric,
    (size_t)10);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUML_LOG_DEBUG("Executed MST");

  /**
   * Perform hierarchical labeling
   */
  size_t n_edges = m - 1;

  rmm::device_uvector<value_idx> children(n_edges * 2, stream);
  rmm::device_uvector<value_t> out_delta(n_edges, stream);
  rmm::device_uvector<value_idx> out_size(n_edges, stream);

  raft::hierarchy::detail::build_dendrogram_host(
    handle, mst_rows.data(), mst_cols.data(), mst_data.data(), n_edges,
    children.data(), out_delta, out_size);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUML_LOG_DEBUG("Executed dendrogram labeling");

  raft::print_device_vector("delta", out_delta.data(), out_delta.size(), std::cout);
  raft::print_device_vector("size", out_size.data(), out_delta.size(), std::cout);

  /**
   * Condense branches of tree according to min cluster size
   */
  detail::Common::CondensedHierarchy<value_idx, value_t> condensed_tree(handle,
                                                                        m);
  detail::Condense::build_condensed_hierarchy(
    handle, children.data(), out_delta.data(), out_size.data(),
    min_cluster_size, m, condensed_tree);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUML_LOG_DEBUG("Executed hierarchy condensing");

  // raft::print_device_vector("condensed parents", condensed_tree.get_parents(), condensed_tree.get_n_edges(), std::cout);
  // raft::print_device_vector("condensed children", condensed_tree.get_children(), condensed_tree.get_n_edges(), std::cout);

  /**
   * Extract labels from stability
   */
  rmm::device_uvector<value_idx> labels(m, stream);
  rmm::device_uvector<value_t> stabilities(m, stream);
  rmm::device_uvector<value_t> probabilities(m, stream);
  detail::Extract::extract_clusters(handle, condensed_tree, m, labels.data(),
                                    stabilities.data(), probabilities.data());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUML_LOG_DEBUG("Executed cluster extraction");
}

};  // end namespace HDBSCAN
};  // end namespace ML