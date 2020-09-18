/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <internals/internals.h>
#include <cuml/common/logger.hpp>
#include <string>

namespace ML {

using namespace ML::Internals;

template<typename value_idx, typename value_t>
struct knn_graph {

  knn_graph(value_idx n_rows_, int n_neighbors_): n_rows(n_rows_), n_neighbors(n_neighbors_){}

  value_idx *knn_indices;
  value_t *knn_dists;

  value_idx n_rows;
  int n_neighbors;
};


template<typename T>
struct umap_inputs_t {
  T *y;
  int n;
  int d;

  umap_inputs_t(T *y_, int n_, int d_): y(y_), n(n_), d(d_) {}

  virtual bool alloc_knn_graph();
};

template<typename T>
struct umap_dense_inputs_t : public umap_inputs_t<T> {
  T *X;

  umap_dense_inputs_t(T *x_, T *y_, int n_, int d_): umap_inputs_t<T>(y_, n_, d_), X(x_) {}

  bool alloc_knn_graph() {
    return true;
  }
};

template<typename value_idx, typename T>
struct umap_sparse_inputs_t : public umap_inputs_t<T> {
  value_idx *indptr;
  value_idx *indices;
  T *data;

  size_t nnz;

  umap_sparse_inputs_t(value_idx *indptr_, value_idx *indices_, T *data_, T *y_, size_t nnz_, int n_, int d_):
    umap_inputs_t<T>(y_, n_, d_), indptr(indptr_), indices(indices_), data(data_), nnz(nnz_) {}

  bool alloc_knn_graph() {
    return true;
  }
};

template<typename T>
struct umap_precomputed_knn_inputs_t : public umap_dense_inputs_t<T> {

  umap_precomputed_knn_inputs_t<T>(int64_t *knn_indices_, T *knn_dists_, T *X_, T *y_,
                                   int n_, int d_): umap_dense_inputs_t<T>(X_, y_, n_, d_),
                                                    knn_indices(knn_indices_), knn_dists(knn_dists_) {}

  int64_t *knn_indices;
  T *knn_dists;

  bool alloc_knn_graph() {
    return false;
  }
};



class UMAPParams {
 public:
  enum MetricType { EUCLIDEAN, CATEGORICAL };

  /**
   *  The number of neighbors to use to approximate geodesic distance.
      Larger numbers induce more global estimates of the manifold that can
      miss finer detail, while smaller values will focus on fine manifold
      structure to the detriment of the larger picture.
   */
  int n_neighbors = 15;

  /**
   * Number of features in the final embedding
   */
  int n_components = 2;

  /**
   * Number of epochs to use in the training of
   * the embedding.
   */
  int n_epochs = 0;

  /**
   * Initial learning rate for the embedding optimization
   */
  float learning_rate = 1.0;

  /**
   *  The effective minimum distance between embedded points. Smaller values
      will result in a more clustered/clumped embedding where nearby points
      on the manifold are drawn closer together, while larger values will
      result on a more even dispersal of points. The value should be set
      relative to the ``spread`` value, which determines the scale at which
      embedded points will be spread out.
   */
  float min_dist = 0.1;

  /**
   *  The effective scale of embedded points. In combination with ``min_dist``
      this determines how clustered/clumped the embedded points are.
   */
  float spread = 1.0;

  /**
   *  Interpolate between (fuzzy) union and intersection as the set operation
      used to combine local fuzzy simplicial sets to obtain a global fuzzy
      simplicial sets. Both fuzzy set operations use the product t-norm.
      The value of this parameter should be between 0.0 and 1.0; a value of
      1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
      intersection.
   */
  float set_op_mix_ratio = 1.0;

  /**
   *  The local connectivity required -- i.e. the number of nearest
      neighbors that should be assumed to be connected at a local level.
      The higher this value the more connected the manifold becomes
      locally. In practice this should be not more than the local intrinsic
      dimension of the manifold.
   */
  float local_connectivity = 1.0;

  /**
   *  Weighting applied to negative samples in low dimensional embedding
      optimization. Values higher than one will result in greater weight
      being given to negative samples.
   */
  float repulsion_strength = 1.0;

  /**
   *  The number of negative samples to select per positive sample
      in the optimization process. Increasing this value will result
      in greater repulsive force being applied, greater optimization
      cost, but slightly more accuracy.
   */
  int negative_sample_rate = 5;

  /**
   *  For transform operations (embedding new points using a trained model_
      this will control how aggressively to search for nearest neighbors.
      Larger values will result in slower performance but more accurate
      nearest neighbor evaluation.
   */
  float transform_queue_size = 4.0;

  /**
   * Control logging level during algorithm execution
   */
  int verbosity = CUML_LEVEL_INFO;

  /**
   *  More specific parameters controlling the embedding. If None these
      values are set automatically as determined by ``min_dist`` and
      ``spread``.
   */
  float a = -1.0;

  /**
   *  More specific parameters controlling the embedding. If None these
      values are set automatically as determined by ``min_dist`` and
      ``spread``.
   */
  float b = -1.0;

  /**
         * Initial learning rate for SGD
         */
  float initial_alpha = 1.0;

  /**
         * Embedding initializer algorithm
         * 0 = random layout
         * 1 = spectral layout
         */
  int init = 1;

  /**
         * The number of nearest neighbors to use to construct the target simplicial
         * set. If set to -1, use the n_neighbors value.
         */
  int target_n_neighbors = -1;

  MetricType target_metric = CATEGORICAL;

  float target_weights = 0.5;

  uint64_t random_state = 0;

  bool multicore_implem = true;

  int optim_batch_size = 0;

  GraphBasedDimRedCallback* callback = nullptr;
};

}  // namespace ML
