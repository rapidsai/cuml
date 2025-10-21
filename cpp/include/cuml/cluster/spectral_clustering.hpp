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
#pragma once

#include <raft/core/handle.hpp>

namespace ML {
namespace SpectralClustering {

/**
 * @brief Spectral clustering parameters
 */
struct SpectralClusteringParams {
  int n_clusters;    // Number of clusters to find
  int n_components;  // Number of eigenvectors to use
  int n_init;        // Number of times to run k-means with different seeds
  int n_neighbors;   // Number of neighbors for kNN graph construction
  uint64_t seed;     // Random seed for reproducibility
};

/**
 * @brief Perform spectral clustering on a precomputed connectivity graph
 *
 * @param[in]  handle          cuML handle
 * @param[in]  params          Parameters for spectral clustering
 * @param[in]  coo_rows        Row indices of the COO sparse matrix
 * @param[in]  coo_cols        Column indices of the COO sparse matrix
 * @param[in]  coo_vals        Values of the COO sparse matrix
 * @param[in]  nnz             Number of non-zero entries in the sparse matrix
 * @param[in]  n_rows          Number of rows in the matrix
 * @param[in]  n_cols          Number of columns in the matrix
 * @param[out] labels          Cluster labels for each sample
 */
void fit_predict(const raft::handle_t& handle,
                 const SpectralClusteringParams& params,
                 const int* coo_rows,
                 const int* coo_cols,
                 const float* coo_vals,
                 int nnz,
                 int n_rows,
                 int n_cols,
                 int* labels);

}  // namespace SpectralClustering
}  // namespace ML
