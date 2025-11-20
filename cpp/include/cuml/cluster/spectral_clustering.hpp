/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace ML {
namespace SpectralClustering {

/**
 * @brief Spectral clustering parameters
 */
struct params {
  /** @brief Number of clusters to find */
  int n_clusters;
  /** @brief Number of eigenvectors to use */
  int n_components;
  /** @brief Number of times to run k-means with different seeds */
  int n_init;
  /** @brief Number of neighbors for kNN graph construction */
  int n_neighbors;
  /** @brief Tolerance for the eigensolver */
  float eigen_tol;
  /** @brief Random seed for reproducibility */
  uint64_t seed;
};

/**
 * @brief Perform spectral clustering on a precomputed connectivity graph
 * using COO sparse matrix view
 *
 * @param[in]  handle              cuML resources handle
 * @param[in]  config              Parameters for spectral clustering
 * @param[in]  connectivity_graph  COO sparse matrix view of the connectivity graph
 * @param[out] labels              Cluster labels for each sample
 */
void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels);

/**
 * @brief Perform spectral clustering on a precomputed connectivity graph
 * using separate vector views for COO components
 *
 * @param[in]  handle   cuML resources handle
 * @param[in]  config   Parameters for spectral clustering
 * @param[in]  rows     Row indices of the COO sparse matrix
 * @param[in]  cols     Column indices of the COO sparse matrix
 * @param[in]  vals     Values of the COO sparse matrix
 * @param[out] labels   Cluster labels for each sample
 */
void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_vector_view<int, int> rows,
                 raft::device_vector_view<int, int> cols,
                 raft::device_vector_view<float, int> vals,
                 raft::device_vector_view<int, int> labels);

}  // namespace SpectralClustering
}  // namespace ML
