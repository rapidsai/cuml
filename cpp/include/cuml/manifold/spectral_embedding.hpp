/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <optional>

namespace cuvs::preprocessing::spectral_embedding {

struct params;

}  // end namespace cuvs::preprocessing::spectral_embedding

namespace ML::SpectralEmbedding {

/**
 * @brief Parameters for spectral embedding algorithm
 */
struct params {
  /** @brief The number of components to reduce the data to. */
  int n_components;

  /** @brief The number of neighbors to use for the nearest neighbors graph. */
  int n_neighbors;

  /** @brief Whether to normalize the Laplacian matrix. */
  bool norm_laplacian;

  /** @brief Whether to drop the first eigenvector. */
  bool drop_first;

  /** @brief Random seed for reproducibility */
  std::optional<uint64_t> seed = std::nullopt;
};

cuvs::preprocessing::spectral_embedding::params to_cuvs(ML::SpectralEmbedding::params& config);

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_matrix_view<float, int, raft::row_major> dataset,
               raft::device_matrix_view<float, int, raft::col_major> embedding);

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_coo_matrix_view<float, int, int, int64_t> connectivity_graph,
               raft::device_matrix_view<float, int, raft::col_major> embedding);

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_vector_view<int, int64_t> rows,
               raft::device_vector_view<int, int64_t> cols,
               raft::device_vector_view<float, int64_t> vals,
               raft::device_matrix_view<float, int, raft::col_major> embedding);

}  // namespace ML::SpectralEmbedding
