/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuml/manifold/tsne.h>
#include "tsne_runner.cuh"

namespace ML {

template <typename tsne_input, typename value_idx, typename value_t>
void _fit(const raft::handle_t &handle, tsne_input &input,
          knn_graph<value_idx, value_t> &k_graph, const value_idx dim,
          const float theta, const float epssq, float perplexity,
          const int perplexity_max_iter, const float perplexity_tol,
          const float early_exaggeration, const float late_exaggeration,
          const int exaggeration_iter, const float min_gain,
          const float pre_learning_rate, const float post_learning_rate,
          const int max_iter, const float min_grad_norm,
          const float pre_momentum, const float post_momentum,
          const long long random_state, int verbosity,
          const bool initialize_embeddings, const bool square_distances,
          TSNE_ALGORITHM algorithm) {
  TSNE_runner<tsne_input, value_idx, value_t> runner(
    handle, input, k_graph, dim, theta, epssq, perplexity, perplexity_max_iter,
    perplexity_tol, early_exaggeration, late_exaggeration, exaggeration_iter,
    min_gain, pre_learning_rate, post_learning_rate, max_iter, min_grad_norm,
    pre_momentum, post_momentum, random_state, verbosity, initialize_embeddings,
    square_distances, algorithm);

  runner.run();
}

void TSNE_fit(const raft::handle_t &handle, float *X, float *Y, int n, int p,
              int64_t *knn_indices, float *knn_dists, const int dim,
              int n_neighbors, const float theta, const float epssq,
              float perplexity, const int perplexity_max_iter,
              const float perplexity_tol, const float early_exaggeration,
              const float late_exaggeration, const int exaggeration_iter,
              const float min_gain, const float pre_learning_rate,
              const float post_learning_rate, const int max_iter,
              const float min_grad_norm, const float pre_momentum,
              const float post_momentum, const long long random_state,
              int verbosity, const bool initialize_embeddings,
              const bool square_distances, TSNE_ALGORITHM algorithm) {
  ASSERT(n > 0 && p > 0 && dim > 0 && n_neighbors > 0 && X != NULL && Y != NULL,
         "Wrong input args");

  manifold_dense_inputs_t<float> input(X, Y, n, p);
  knn_graph<int64_t, float> k_graph(n, n_neighbors, knn_indices, knn_dists);

  _fit<manifold_dense_inputs_t<float>, knn_indices_dense_t, float>(
    handle, input, k_graph, dim, theta, epssq, perplexity, perplexity_max_iter,
    perplexity_tol, early_exaggeration, late_exaggeration, exaggeration_iter,
    min_gain, pre_learning_rate, post_learning_rate, max_iter, min_grad_norm,
    pre_momentum, post_momentum, random_state, verbosity, initialize_embeddings,
    square_distances, algorithm);
}

void TSNE_fit_sparse(const raft::handle_t &handle, int *indptr, int *indices,
                     float *data, float *Y, int nnz, int n, int p,
                     int *knn_indices, float *knn_dists, const int dim,
                     int n_neighbors, const float theta, const float epssq,
                     float perplexity, const int perplexity_max_iter,
                     const float perplexity_tol, const float early_exaggeration,
                     const float late_exaggeration, const int exaggeration_iter,
                     const float min_gain, const float pre_learning_rate,
                     const float post_learning_rate, const int max_iter,
                     const float min_grad_norm, const float pre_momentum,
                     const float post_momentum, const long long random_state,
                     int verbosity, const bool initialize_embeddings,
                     const bool square_distances, TSNE_ALGORITHM algorithm) {
  ASSERT(n > 0 && p > 0 && dim > 0 && n_neighbors > 0 && indptr != NULL &&
           indices != NULL && data != NULL && Y != NULL,
         "Wrong input args");

  manifold_sparse_inputs_t<int, float> input(indptr, indices, data, Y, nnz, n,
                                             p);
  knn_graph<int, float> k_graph(n, n_neighbors, knn_indices, knn_dists);

  _fit<manifold_sparse_inputs_t<int, float>, knn_indices_sparse_t, float>(
    handle, input, k_graph, dim, theta, epssq, perplexity, perplexity_max_iter,
    perplexity_tol, early_exaggeration, late_exaggeration, exaggeration_iter,
    min_gain, pre_learning_rate, post_learning_rate, max_iter, min_grad_norm,
    pre_momentum, post_momentum, random_state, verbosity, initialize_embeddings,
    square_distances, algorithm);
}

}  // namespace ML
