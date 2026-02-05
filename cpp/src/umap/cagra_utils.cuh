/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <cuml/manifold/common.hpp>
#include <cuml/manifold/umap.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/core/handle.hpp>
#include <raft/linalg/unary_op.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/cagra_optimize.hpp>

#include <memory>

namespace UMAPAlgo {
namespace CagraUtils {

/**
 * Build a CAGRA index from a KNN graph and dataset.
 *
 * @param handle RAFT handle
 * @param knn_indices Device pointer to KNN indices (n x n_neighbors)
 * @param n Number of data points
 * @param n_neighbors Number of neighbors in the KNN graph
 * @param X Dataset pointer (device or host memory)
 * @param d Dataset dimensionality
 * @param metric Distance metric to use
 * @param stream CUDA stream
 * @return unique_ptr to the built CAGRA index
 */
template <typename value_idx, typename value_t>
std::unique_ptr<ML::cagra_index_t> build_cagra_index(const raft::handle_t& handle,
                                                     value_idx* knn_indices,
                                                     int n,
                                                     int n_neighbors,
                                                     value_t* X,
                                                     int d,
                                                     ML::distance::DistanceType metric,
                                                     cudaStream_t stream)
{
  using nnz_t           = int64_t;
  nnz_t n_x_n_neighbors = static_cast<nnz_t>(n) * n_neighbors;

  CUML_LOG_DEBUG("Building CAGRA index from KNN graph");

  // Validate metric - CAGRA only supports L2, InnerProduct, and Cosine
  RAFT_EXPECTS(metric == ML::distance::DistanceType::L2Expanded ||
                 metric == ML::distance::DistanceType::L2SqrtExpanded ||
                 metric == ML::distance::DistanceType::L2Unexpanded ||
                 metric == ML::distance::DistanceType::L2SqrtUnexpanded ||
                 metric == ML::distance::DistanceType::InnerProduct ||
                 metric == ML::distance::DistanceType::CosineExpanded,
               "fast_transform (CAGRA) only supports L2, InnerProduct, and Cosine metrics. "
               "Use fast_transform=False for other metrics.");

  // Convert value_idx indices to uint32_t on device
  rmm::device_uvector<uint32_t> knn_indices_u32(n_x_n_neighbors, stream);
  raft::linalg::unary_op(
    handle,
    raft::make_device_matrix_view<const value_idx, int64_t>(knn_indices, n, n_neighbors),
    raft::make_device_matrix_view<uint32_t, int64_t>(knn_indices_u32.data(), n, n_neighbors),
    [] __device__(value_idx val) { return static_cast<uint32_t>(val); });

  // Copy to host
  auto h_knn_indices = raft::make_host_matrix<uint32_t, int64_t>(n, n_neighbors);
  raft::copy(h_knn_indices.data_handle(), knn_indices_u32.data(), n_x_n_neighbors, stream);
  handle.sync_stream(stream);

  // Optimize graph for CAGRA
  size_t optimized_degree = std::max(static_cast<size_t>(16), static_cast<size_t>(n_neighbors) / 2);
  optimized_degree        = std::min(optimized_degree, static_cast<size_t>(n_neighbors));
  auto h_optimized_graph  = raft::make_host_matrix<uint32_t, int64_t>(n, optimized_degree);

  CUML_LOG_DEBUG(
    "Optimizing KNN graph: input degree=%d, output degree=%zu", n_neighbors, optimized_degree);
  cuvs::neighbors::cagra::helpers::optimize(handle, h_knn_indices.view(), h_optimized_graph.view());

  // Build CAGRA index
  auto cagra_metric = static_cast<cuvs::distance::DistanceType>(metric);

  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, X));
  bool data_on_device = attr.type == cudaMemoryTypeDevice;

  std::unique_ptr<ML::cagra_index_t> cagra_index;
  if (data_on_device) {
    auto dataset_view = raft::make_device_matrix_view<const value_t, int64_t>(X, n, d);
    cagra_index       = std::make_unique<ML::cagra_index_t>(
      handle, cagra_metric, dataset_view, raft::make_const_mdspan(h_optimized_graph.view()));
  } else {
    auto dataset_view = raft::make_host_matrix_view<const value_t, int64_t>(X, n, d);
    cagra_index       = std::make_unique<ML::cagra_index_t>(
      handle, cagra_metric, dataset_view, raft::make_const_mdspan(h_optimized_graph.view()));
  }

  CUML_LOG_DEBUG("CAGRA index built successfully, size=%lu, dim=%u, graph_degree=%u",
                 cagra_index->size(),
                 cagra_index->dim(),
                 cagra_index->graph_degree());

  return cagra_index;
}

/**
 * Search the CAGRA index for nearest neighbors.
 *
 * @param handle RAFT handle
 * @param cagra_index The CAGRA index to search
 * @param queries Device pointer to query points (n_queries x d)
 * @param n_queries Number of query points
 * @param d Query dimensionality
 * @param k Number of neighbors to find
 * @param out_indices Output device pointer for neighbor indices (n_queries x k)
 * @param out_distances Output device pointer for neighbor distances (n_queries x k)
 * @param stream CUDA stream
 */
template <typename value_idx, typename value_t>
void search_cagra_index(const raft::handle_t& handle,
                        const ML::cagra_index_t& cagra_index,
                        const value_t* queries,
                        int n_queries,
                        int d,
                        int k,
                        value_idx* out_indices,
                        value_t* out_distances,
                        cudaStream_t stream)
{
  CUML_LOG_DEBUG("Using CAGRA index for KNN search");

  // Allocate temporary buffers for CAGRA search output (uint32_t indices)
  rmm::device_uvector<uint32_t> cagra_indices(static_cast<size_t>(n_queries) * k, stream);
  rmm::device_uvector<value_t> cagra_distances(static_cast<size_t>(n_queries) * k, stream);

  // Create views for CAGRA search
  auto queries_view = raft::make_device_matrix_view<const value_t, int64_t>(queries, n_queries, d);
  auto indices_view =
    raft::make_device_matrix_view<uint32_t, int64_t>(cagra_indices.data(), n_queries, k);
  auto distances_view =
    raft::make_device_matrix_view<value_t, int64_t>(cagra_distances.data(), n_queries, k);

  // Perform CAGRA search
  cuvs::neighbors::cagra::search_params search_params;
  cuvs::neighbors::cagra::search(
    handle, search_params, cagra_index, queries_view, indices_view, distances_view);

  // Convert uint32_t indices to value_idx
  raft::linalg::unary_op(
    handle,
    raft::make_device_matrix_view<const uint32_t, int64_t>(cagra_indices.data(), n_queries, k),
    raft::make_device_matrix_view<value_idx, int64_t>(out_indices, n_queries, k),
    [] __device__(uint32_t val) { return static_cast<value_idx>(val); });

  // Copy distances
  raft::copy(out_distances, cagra_distances.data(), n_queries * k, stream);
}

}  // namespace CagraUtils
}  // namespace UMAPAlgo
