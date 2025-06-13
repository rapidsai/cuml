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

#include <cuml/common/callback.hpp>
#include <cuml/common/distance_type.hpp>
#include <cuml/common/logger.hpp>

namespace ML {

namespace graph_build_params {

/**
 * Arguments for using nn descent as the knn build algorithm.
 * graph_degree must be larger than or equal to n_neighbors.
 * Increasing graph_degree and max_iterations may result in better accuracy.
 * Smaller termination threshold means stricter convergence criteria for nn descent and may take
 * longer to converge.
 */
struct nn_descent_params_umap {
  // not directly using cuvs::neighbors::nn_descent::index_params to distinguish UMAP-exposed NN
  // Descent parameters
  size_t graph_degree              = 64;
  size_t intermediate_graph_degree = 128;
  size_t max_iterations            = 20;
  float termination_threshold      = 0.0001;
};

/**
 * Parameters for knn graph building in UMAP.
 * [Hint1]: the ratio of overlap_factor / n_clusters determines device memory usage.
 * Approximately (overlap_factor / n_clusters) * num_rows_in_entire_data number of rows will be
 * put on device memory at once. E.g. between (overlap_factor / n_clusters) = 2/10 and 2/20, the
 * latter will use less device memory.
 * [Hint2]: larger overlap_factor results in better accuracy
 * of the final all-neighbors knn graph. E.g. While using similar amount of device memory,
 * (overlap_factor / n_clusters) = 4/20 will have better accuracy than 2/10 at the cost of
 * performance.
 * [Hint3]: for overlap_factor, start with 2, and gradually increase (2->3->4 ...)
 * for better accuracy
 * [Hint4]: for n_clusters, start with 4, and gradually increase(4->8->16 ...)
 * for less GPU memory usage. This is independent from overlap_factor as long as
 * overlap_factor < n_clusters
 */
struct graph_build_params {
  /**
   * Number of clusters each data point is assigned to. Only valid when n_clusters > 1.
   */
  size_t overlap_factor = 2;
  /**
   * Number of clusters to split the data into when building the knn graph. Increasing this will use
   * less device memory at the cost of accuracy. When using n_clusters > 1, is is required that the
   * data is put on host (refer to data_on_host argument for fit_transform). The default value
   * (n_clusters=1) will place the entire data on device memory.
   */
  size_t n_clusters = 1;
  nn_descent_params_umap nn_descent_params;
};
}  // namespace graph_build_params

class UMAPParams {
 public:
  enum MetricType { EUCLIDEAN, CATEGORICAL };
  enum graph_build_algo { BRUTE_FORCE_KNN, NN_DESCENT };

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
  rapids_logger::level_enum verbosity = rapids_logger::level_enum::info;

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
   * KNN graph build algorithm
   */
  graph_build_algo build_algo = graph_build_algo::BRUTE_FORCE_KNN;

  graph_build_params::graph_build_params build_params;

  /**
   * The number of nearest neighbors to use to construct the target simplicial
   * set. If set to -1, use the n_neighbors value.
   */
  int target_n_neighbors = -1;

  MetricType target_metric = CATEGORICAL;

  float target_weight = 0.5;

  uint64_t random_state = 0;

  /**
   *  Whether should we use deterministic algorithm.  This should be set to true if
      random_state is provided, otherwise it's false.  When it's true, cuml will have
      higher memory usage but produce stable numeric output.
   */
  bool deterministic = true;

  ML::distance::DistanceType metric = ML::distance::DistanceType::L2SqrtExpanded;

  float p = 2.0;

  Internals::GraphBasedDimRedCallback* callback = nullptr;
};

}  // namespace ML
