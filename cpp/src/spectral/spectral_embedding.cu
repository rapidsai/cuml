/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuml/manifold/spectral_embedding.hpp>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/preprocessing/spectral_embedding.hpp>

namespace ML::SpectralEmbedding {

cuvs::preprocessing::spectral_embedding::params to_cuvs(ML::SpectralEmbedding::params& config)
{
  cuvs::preprocessing::spectral_embedding::params params;

  params.n_components   = config.n_components;
  params.n_neighbors    = config.n_neighbors;
  params.norm_laplacian = config.norm_laplacian;
  params.drop_first     = config.drop_first;
  params.seed           = config.seed;

  return params;
}

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_matrix_view<float, int, raft::row_major> dataset,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  cuvs::preprocessing::spectral_embedding::transform(handle, to_cuvs(config), dataset, embedding);
}

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  cuvs::preprocessing::spectral_embedding::transform(
    handle, to_cuvs(config), connectivity_graph, embedding);
}

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_vector_view<int, int> rows,
               raft::device_vector_view<int, int> cols,
               raft::device_vector_view<float, int> vals,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  auto connectivity_graph_view = raft::make_device_coo_matrix_view<float, int, int, int>(
    vals.data_handle(),
    raft::make_device_coordinate_structure_view<int, int, int>(rows.data_handle(),
                                                               cols.data_handle(),
                                                               embedding.extent(0),
                                                               embedding.extent(0),
                                                               vals.size()));

  ML::SpectralEmbedding::transform(handle, config, connectivity_graph_view, embedding);
}

}  // namespace ML::SpectralEmbedding
