/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <spectral/spectral_embedding.hpp>

namespace ML::SpectralEmbedding {

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_matrix_view<float, int, raft::row_major> dataset,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  cuvs::preprocessing::spectral_embedding::transform(handle, to_cuvs(config), dataset, embedding);
}

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_coo_matrix_view<float, int, int, int64_t> connectivity_graph,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  cuvs::preprocessing::spectral_embedding::transform(
    handle, to_cuvs(config), connectivity_graph, embedding);
}

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_vector_view<int, int64_t> rows,
               raft::device_vector_view<int, int64_t> cols,
               raft::device_vector_view<float, int64_t> vals,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  auto connectivity_graph_view = raft::make_device_coo_matrix_view<float, int, int, int64_t>(
    vals.data_handle(),
    raft::make_device_coordinate_structure_view<int, int, int64_t>(rows.data_handle(),
                                                                   cols.data_handle(),
                                                                   embedding.extent(0),
                                                                   embedding.extent(0),
                                                                   vals.size()));

  ML::SpectralEmbedding::transform(handle, config, connectivity_graph_view, embedding);
}

}  // namespace ML::SpectralEmbedding
