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

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

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
  uint64_t seed;
};

cuvs::preprocessing::spectral_embedding::params to_cuvs(ML::SpectralEmbedding::params& config);

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_matrix_view<float, int, raft::row_major> dataset,
               raft::device_matrix_view<float, int, raft::col_major> embedding);

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
               raft::device_matrix_view<float, int, raft::col_major> embedding);

void transform(raft::resources const& handle,
               ML::SpectralEmbedding::params config,
               raft::device_vector_view<int, int> rows,
               raft::device_vector_view<int, int> cols,
               raft::device_vector_view<float, int> vals,
               raft::device_matrix_view<float, int, raft::col_major> embedding);

}  // namespace ML::SpectralEmbedding
