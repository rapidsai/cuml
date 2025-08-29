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

#include <cuml/common/distance_type.hpp>
#include <cuml/common/logger.hpp>

namespace raft {
class handle_t;
}

namespace ML {

enum TSNE_ALGORITHM { EXACT, BARNES_HUT, FFT };

enum TSNE_INIT { RANDOM, PCA };

struct TSNEParams {
  // Number of output dimensions for embeddings Y.
  int dim = 2;

  // Number of nearest neighbors used.
  int n_neighbors = 1023;

  // Float between 0 and 1. Tradeoff for speed (0) vs accuracy (1).
  // (Barnes-Hut only.)
  float theta = 0.5f;

  // A tiny jitter to promote numerical stability. (Barnes-Hut only.)
  float epssq = 0.0025;

  // How many nearest neighbors are used during construction of Pij.
  float perplexity = 50.0f;

  // Number of iterations used to construct Pij.
  int perplexity_max_iter = 100;

  // The small tolerance used for Pij to ensure numerical stability.
  float perplexity_tol = 1e-5;

  // How much pressure to apply to clusters to spread out
  // during the exaggeration phase.
  float early_exaggeration = 12.0f;

  // How much pressure to apply to clusters to
  // spread out after the exaggeration phase. (FIT-SNE only)
  float late_exaggeration = 1.0f;

  // How many iterations you want the early pressure to run for.
  // If late exaggeration is used, it will be applied to all iterations
  // that remain after this number of iterations.
  int exaggeration_iter = 250;

  // Rounds up small gradient updates. (Barnes-Hut and Exact only.)
  float min_gain = 0.01f;

  // The learning rate during exaggeration phase.
  float pre_learning_rate = 200.0f;

  // The learning rate after exaggeration phase.
  float post_learning_rate = 500.0f;

  // The maximum number of iterations TSNE should run for.
  int max_iter = 1000;

  // The smallest gradient norm TSNE should terminate on.
  // (Exact only; ignored for others.)
  float min_grad_norm = 1e-7;

  // The momentum used during the exaggeration phase.
  float pre_momentum = 0.5;

  // The momentum used after the exaggeration phase.
  float post_momentum = 0.8;

  // Set this to -1 for pure random initializations or >= 0 for
  // reproducible outputs. This sets random seed correctly, but there
  // may still be some variance due to the parallel nature of this algorithm.
  long long random_state = -1;

  // verbosity level for logging messages during execution
  rapids_logger::level_enum verbosity = rapids_logger::level_enum::info;

  // Embedding initializer algorithm
  TSNE_INIT init = TSNE_INIT::RANDOM;

  // When this is set to true, the distances from the knn graph will
  // always be squared before computing conditional probabilities, even if
  // the knn graph is passed in explicitly. This is to better match the
  // behavior of Scikit-learn's T-SNE.
  bool square_distances = true;

  // Distance metric to use.
  ML::distance::DistanceType metric = ML::distance::DistanceType::L2SqrtExpanded;

  // Value of p for Minkowski distance
  float p = 2.0;

  // Which implementation algorithm to use.
  TSNE_ALGORITHM algorithm = TSNE_ALGORITHM::FFT;
};

/**
 * @brief Dimensionality reduction via TSNE using Barnes-Hut, Fourier Interpolation, or naive
 * methods. or brute force O(N^2).
 *
 * @param[in]  handle              The GPU handle.
 * @param[in]  X                   The row-major dataset in device memory.
 * @param[out] Y                   The column-major final embedding in device memory
 * @param[in]  n                   Number of rows in data X.
 * @param[in]  p                   Number of columns in data X.
 * @param[in]  knn_indices         Array containing nearest neighbors indices.
 * @param[in]  knn_dists           Array containing nearest neighbors distances.
 * @param[in]  params              Parameters for TSNE model
 * @param[out] kl_div              (optional) KL divergence output
 * @param[out] n_iter              (optional) The number of iterations TSNE ran for.
 *
 * The CUDA implementation is derived from the excellent CannyLabs open source
 * implementation here: https://github.com/CannyLab/tsne-cuda/. The CannyLabs
 * code is licensed according to the conditions in
 * cuml/cpp/src/tsne/cannylabs_tsne_license.txt. A full description of their
 * approach is available in their article t-SNE-CUDA: GPU-Accelerated t-SNE and
 * its Applications to Modern Data (https://arxiv.org/abs/1807.11824).
 */
void TSNE_fit(const raft::handle_t& handle,
              float* X,
              float* Y,
              int n,
              int p,
              int64_t* knn_indices,
              float* knn_dists,
              TSNEParams& params,
              float* kl_div = nullptr,
              int* n_iter   = nullptr);

/**
 * @brief Dimensionality reduction via TSNE using either Barnes Hut O(NlogN)
 *       or brute force O(N^2).
 *
 * @param[in]  handle              The GPU handle.
 * @param[in]  indptr              indptr of CSR dataset.
 * @param[in]  indices             indices of CSR dataset.
 * @param[in]  data                data of CSR dataset.
 * @param[out] Y                   The final embedding.
 * @param[in]  nnz                 The number of non-zero entries in the CSR.
 * @param[in]  n                   Number of rows in data X.
 * @param[in]  p                   Number of columns in data X.
 * @param[in]  knn_indices         Array containing nearest neighbors indices.
 * @param[in]  knn_dists           Array containing nearest neighbors distances.
 * @param[in]  params              Parameters for TSNE model
 * @param[out] kl_div              (optional) KL divergence output
 * @param[out] n_iter              (optional) The number of iterations TSNE ran for.
 *
 * The CUDA implementation is derived from the excellent CannyLabs open source
 * implementation here: https://github.com/CannyLab/tsne-cuda/. The CannyLabs
 * code is licensed according to the conditions in
 * cuml/cpp/src/tsne/cannylabs_tsne_license.txt. A full description of their
 * approach is available in their article t-SNE-CUDA: GPU-Accelerated t-SNE and
 * its Applications to Modern Data (https://arxiv.org/abs/1807.11824).
 */
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
                     float* kl_div = nullptr,
                     int* n_iter   = nullptr);

}  // namespace ML
