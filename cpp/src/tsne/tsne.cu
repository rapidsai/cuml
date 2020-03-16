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

#include <cuml/manifold/tsne.h>
#include <cuml/common/logger.hpp>
#include "../../src_prims/utils.h"
#include "distances.h"
#include "exact_kernels.h"
#include "utils.h"

#include "barnes_hut.h"
#include "exact_tsne.h"

namespace ML {

/**
 * @brief Dimensionality reduction via TSNE using either Barnes Hut O(NlogN) or brute force O(N^2).
 * @input param handle: The GPU handle.
 * @input param X: The dataset you want to apply TSNE on.
 * @output param Y: The final embedding. Will overwrite this internally.
 * @input param n: Number of rows in data X.
 * @input param p: Number of columns in data X.
 * @input param dim: Number of output dimensions for embeddings Y.
 * @input param n_neighbors: Number of nearest neighbors used.
 * @input param theta: Float between 0 and 1. Tradeoff for speed (0) vs accuracy (1) for Barnes Hut only.
 * @input param epssq: A tiny jitter to promote numerical stability.
 * @input param perplexity: How many nearest neighbors are used during the construction of Pij.
 * @input param perplexity_max_iter: Number of iterations used to construct Pij.
 * @input param perplexity_tol: The small tolerance used for Pij to ensure numerical stability.
 * @input param early_exaggeration: How much early pressure you want the clusters in TSNE to spread out more.
 * @input param exaggeration_iter: How many iterations you want the early pressure to run for.
 * @input param min_gain: Rounds up small gradient updates.
 * @input param pre_learning_rate: The learning rate during the exaggeration phase.
 * @input param post_learning_rate: The learning rate after the exaggeration phase.
 * @input param max_iter: The maximum number of iterations TSNE should run for.
 * @input param min_grad_norm: The smallest gradient norm TSNE should terminate on.
 * @input param pre_momentum: The momentum used during the exaggeration phase.
 * @input param post_momentum: The momentum used after the exaggeration phase.
 * @input param random_state: Set this to -1 for pure random intializations or >= 0 for reproducible outputs.
 * @input param verbose: Whether to print error messages or not.
 * @input param intialize_embeddings: Whether to overwrite the current Y vector with random noise.
 * @input param barnes_hut: Whether to use the fast Barnes Hut or use the slower exact version.
 */
void TSNE_fit(const cumlHandle &handle, const float *X, float *Y, const int n,
              const int p, const int dim, int n_neighbors, const float theta,
              const float epssq, float perplexity,
              const int perplexity_max_iter, const float perplexity_tol,
              const float early_exaggeration, const int exaggeration_iter,
              const float min_gain, const float pre_learning_rate,
              const float post_learning_rate, const int max_iter,
              const float min_grad_norm, const float pre_momentum,
              const float post_momentum, const long long random_state,
              const bool verbose, const bool intialize_embeddings,
              bool barnes_hut) {
  ASSERT(n > 0 && p > 0 && dim > 0 && n_neighbors > 0 && X != NULL && Y != NULL,
         "Wrong input args");
  ML::Logger::get().setLevel(verbose ? CUML_LEVEL_INFO : CUML_LEVEL_WARN);
  if (dim > 2 and barnes_hut) {
    barnes_hut = false;
    CUML_LOG_WARN(
      "Barnes Hut only works for dim == 2. Switching to exact solution.");
  }
  if (n_neighbors > n) n_neighbors = n;
  if (n_neighbors > 1023) {
    CUML_LOG_WARN("FAISS only supports maximum n_neighbors = 1023.");
    n_neighbors = 1023;
  }
  // Perplexity must be less than number of datapoints
  // "How to Use t-SNE Effectively" https://distill.pub/2016/misread-tsne/
  if (perplexity > n) perplexity = n;

  CUML_LOG_INFO("Data size = (%d, %d) with dim = %d perplexity = %f", n, p, dim,
                perplexity);
  if (perplexity < 5 or perplexity > 50)
    CUML_LOG_WARN(
      "Perplexity should be within ranges (5, 50). Your results might be a"
      " bit strange...");
  if (n_neighbors < perplexity * 3.0f)
    CUML_LOG_WARN(
      "# of Nearest Neighbors should be at least 3 * perplexity. Your results"
      " might be a bit strange...");

  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  START_TIMER;
  //---------------------------------------------------
  // Get distances
  CUML_LOG_INFO("Getting distances.");
  float *distances =
    (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);
  long *indices =
    (long *)d_alloc->allocate(sizeof(long) * n * n_neighbors, stream);
  TSNE::get_distances(X, n, p, indices, distances, n_neighbors, d_alloc,
                      stream);
  //---------------------------------------------------
  END_TIMER(DistancesTime);

  START_TIMER;
  //---------------------------------------------------
  // Normalize distances
  CUML_LOG_INFO("Now normalizing distances so exp(D) doesn't explode.");
  TSNE::normalize_distances(n, distances, n_neighbors, stream);
  //---------------------------------------------------
  END_TIMER(NormalizeTime);

  START_TIMER;
  //---------------------------------------------------
  // Optimal perplexity
  CUML_LOG_INFO("Searching for optimal perplexity via bisection search.");
  float *P =
    (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);
  const float P_sum =
    TSNE::perplexity_search(distances, P, perplexity, perplexity_max_iter,
                            perplexity_tol, n, n_neighbors, handle);
  d_alloc->deallocate(distances, sizeof(float) * n * n_neighbors, stream);
  CUML_LOG_INFO("Perplexity sum = %f", P_sum);
  //---------------------------------------------------
  END_TIMER(PerplexityTime);

  START_TIMER;
  //---------------------------------------------------
  // Convert data to COO layout
  MLCommon::Sparse::COO<float> COO_Matrix(d_alloc, stream);
  TSNE::symmetrize_perplexity(P, indices, n, n_neighbors, P_sum,
                              early_exaggeration, &COO_Matrix, stream, handle);
  const int NNZ = COO_Matrix.nnz;
  float *VAL = COO_Matrix.vals();
  const int *COL = COO_Matrix.cols();
  const int *ROW = COO_Matrix.rows();
  //---------------------------------------------------
  END_TIMER(SymmetrizeTime);

  if (barnes_hut) {
    TSNE::Barnes_Hut(VAL, COL, ROW, NNZ, handle, Y, n, theta, epssq,
                     early_exaggeration, exaggeration_iter, min_gain,
                     pre_learning_rate, post_learning_rate, max_iter,
                     min_grad_norm, pre_momentum, post_momentum, random_state);
  } else {
    TSNE::Exact_TSNE(VAL, COL, ROW, NNZ, handle, Y, n, dim, early_exaggeration,
                     exaggeration_iter, min_gain, pre_learning_rate,
                     post_learning_rate, max_iter, min_grad_norm, pre_momentum,
                     post_momentum, random_state, intialize_embeddings);
  }
}

}  // namespace ML
