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

#include <cuml/cluster/spectral_clustering.hpp>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/cluster/spectral.hpp>

namespace ML {
namespace SpectralClustering {

void fit_predict(const raft::handle_t& handle,
                 const SpectralClusteringParams& params,
                 const int* coo_rows,
                 const int* coo_cols,
                 const float* coo_vals,
                 int nnz,
                 int n_rows,
                 int n_cols,
                 int* labels)
{
  cuvs::cluster::spectral::params cuvs_params;
  cuvs_params.n_clusters   = params.n_clusters;
  cuvs_params.n_components = params.n_components;
  cuvs_params.n_init       = params.n_init;
  cuvs_params.n_neighbors  = params.n_neighbors;
  cuvs_params.seed         = params.seed;

  auto structure = raft::make_device_coordinate_structure_view<int, int, int>(
    const_cast<int*>(coo_rows), const_cast<int*>(coo_cols), n_rows, n_cols, nnz);

  auto coo_matrix = raft::make_device_coo_matrix_view<float, int, int, int>(
    const_cast<float*>(coo_vals), structure);

  auto labels_view = raft::make_device_vector_view<int, int>(labels, n_rows);

  cuvs::cluster::spectral::fit_predict(handle, cuvs_params, coo_matrix, labels_view);
}

}  // namespace SpectralClustering
}  // namespace ML
