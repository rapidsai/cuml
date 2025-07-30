/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuml/manifold/spectral_embedding.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/sparse/coo.hpp>

#include <stdint.h>

namespace UMAPAlgo {

namespace InitEmbed {

namespace SpectralInit {

using namespace ML;

/**
 * Performs a spectral layout initialization
 */
template <typename T, typename nnz_t>
void launcher(const raft::handle_t& handle,
              nnz_t n,
              int d,
              raft::sparse::COO<float>* coo,
              UMAPParams* params,
              T* embedding)
{
  cudaStream_t stream = handle.get_stream();

  ASSERT(n > static_cast<nnz_t>(params->n_components),
         "Spectral layout requires n_samples > n_components");

  auto connectivity_graph_view = raft::make_device_coo_matrix_view<float, int, int, int>(
    coo->vals(),
    raft::make_device_coordinate_structure_view<int, int, int>(
      coo->rows(), coo->cols(), n, n, coo->nnz));

  ML::SpectralEmbedding::params spectral_params;
  spectral_params.n_neighbors    = params->n_neighbors;
  spectral_params.norm_laplacian = true;
  spectral_params.drop_first     = true;
  spectral_params.seed           = params->random_state;
  spectral_params.n_components =
    spectral_params.drop_first ? params->n_components + 1 : params->n_components;

  auto tmp_embedding      = raft::make_device_vector<float, int>(handle, n * params->n_components);
  auto tmp_embedding_view = raft::make_device_matrix_view<float, int, raft::col_major>(
    tmp_embedding.data_handle(), n, params->n_components);

  ML::SpectralEmbedding::transform(
    handle, spectral_params, connectivity_graph_view, tmp_embedding_view);

  raft::linalg::transpose(
    handle, tmp_embedding.data_handle(), embedding, n, params->n_components, stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace SpectralInit
}  // namespace InitEmbed
};  // namespace UMAPAlgo
