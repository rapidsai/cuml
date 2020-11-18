/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuml/cuml.hpp>
#include "umapparams.h"

namespace ML {

void transform(const raft::handle_t &handle, float *X, int n, int d,
               int64_t *knn_indices, float *knn_dists, float *orig_X,
               int orig_n, float *embedding, int embedding_n,
               UMAPParams *params, float *transformed);

void find_ab(const raft::handle_t &handle, UMAPParams *params);

void fit(const raft::handle_t &handle,
         float *X,  // input matrix
         float *y,  // labels
         int n, int d, int64_t *knn_indices, float *knn_dists,
         UMAPParams *params, float *embeddings);

void fit(const raft::handle_t &handle,
         float *X,  // input matrix
         int n,     // rows
         int d,     // cols
         int64_t *knn_indices, float *knn_dists, UMAPParams *params,
         float *embeddings);

class UMAP_API {
  float *orig_X;
  int orig_n;
  raft::handle_t *handle;
  UMAPParams *params;

 public:
  UMAP_API(const raft::handle_t &handle, UMAPParams *params);
  ~UMAP_API();

  /**
   * Fits an unsupervised UMAP model
   * @param X
   *        pointer to an array in row-major format (note: this will be col-major soon)
   * @param n
   *        n_samples in X
   * @param d
   *        d_features in X
   * @param knn_indices
   *        an array containing the n_neighbors nearest neighors indices for each sample
   * @param knn_dists
   *        an array containing the n_neighbors nearest neighors distances for each sample
   * @param embeddings
   *        an array to return the output embeddings of size (n_samples, n_components)
   */
  void fit(float *X, int n, int d, int64_t *knn_indices, float *knn_dists,
           float *embeddings);

  /**
   * Fits a supervised UMAP model
   * @param X
   *        pointer to an array in row-major format (note: this will be col-major soon)
   * @param y
   *        pointer to an array of labels, shape=n_samples
   * @param n
   *        n_samples in X
   * @param d
   *        d_features in X
   * @param knn_indices
   *        an array containing the n_neighbors nearest neighors indices for each sample
   * @param knn_dists
   *        an array containing the n_neighbors nearest neighors distances for each sample
   * @param embeddings
   *        an array to return the output embeddings of size (n_samples, n_components)
   */
  void fit(float *X, float *y, int n, int d, int64_t *knn_indices,
           float *knn_dists, float *embeddings);

  /**
   * Project a set of X vectors into the embedding space.
   * @param X
   *        pointer to an array in row-major format (note: this will be col-major soon)
   * @param n
   *        n_samples in X
   * @param d
   *        d_features in X
   * @param knn_indices
   *        an array containing the n_neighbors nearest neighors indices for each sample
   * @param knn_dists
   *        an array containing the n_neighbors nearest neighors distances for each sample
   * @param embedding
   *        pointer to embedding array of size (embedding_n, n_components) that has been created with fit()
   * @param embedding_n
   *        n_samples in embedding array
   * @param out
   *        pointer to array for storing output embeddings (n, n_components)
   */
  void transform(float *X, int n, int d, int64_t *knn_indices, float *knn_dists,
                 float *embedding, int embedding_n, float *out);

  /**
   * Get the UMAPParams instance
   */
  UMAPParams *get_params();
};
}  // namespace ML
