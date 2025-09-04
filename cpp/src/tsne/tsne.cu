/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "tsne_runner.cuh"

#include <cuml/manifold/tsne.h>

#include <raft/core/handle.hpp>

#include <cuvs/distance/distance.hpp>

#include <utility>

namespace ML {
template <typename tsne_input, typename value_idx, typename value_t>
std::pair<float, int> _fit(const raft::handle_t& handle,
                           tsne_input& input,
                           knn_graph<value_idx, value_t>& k_graph,
                           TSNEParams& params)
{
  TSNE_runner<tsne_input, value_idx, value_t> runner(handle, input, k_graph, params);

  return runner.run();
}

void TSNE_fit(const raft::handle_t& handle,
              float* X,
              float* Y,
              int n,
              int p,
              int64_t* knn_indices,
              float* knn_dists,
              TSNEParams& params,
              float* kl_div,
              int* n_iter)
{
  ASSERT(n > 0 && p > 0 && params.dim > 0 && params.n_neighbors > 0 && X != NULL && Y != NULL,
         "Wrong input args");

  manifold_dense_inputs_t<float> input(X, Y, n, p);
  knn_graph<int64_t, float> k_graph(n, params.n_neighbors, knn_indices, knn_dists);

  auto stats = _fit<manifold_dense_inputs_t<float>, knn_indices_dense_t, float>(
    handle, input, k_graph, params);

  if (kl_div) { *kl_div = stats.first; }
  if (n_iter) { *n_iter = stats.second; }
}

void TSNE_fit_sparse(const raft::handle_t& handle,
                     int* indptr,
                     int* indices,
                     float* data,
                     float* Y,
                     int nnz,
                     int n,
                     int p,
                     int* knn_indices,
                     float* knn_dists,
                     TSNEParams& params,
                     float* kl_div,
                     int* n_iter)
{
  ASSERT(n > 0 && p > 0 && params.dim > 0 && params.n_neighbors > 0 && indptr != NULL &&
           indices != NULL && data != NULL && Y != NULL,
         "Wrong input args");

  manifold_sparse_inputs_t<int, float> input(indptr, indices, data, Y, nnz, n, p);
  knn_graph<int, float> k_graph(n, params.n_neighbors, knn_indices, knn_dists);

  auto stats = _fit<manifold_sparse_inputs_t<int, float>, knn_indices_sparse_t, float>(
    handle, input, k_graph, params);

  if (kl_div) { *kl_div = stats.first; }
  if (n_iter) { *n_iter = stats.second; }
}

}  // namespace ML
