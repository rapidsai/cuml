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
#include <raft/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <cuml/common/logger.hpp>
#include "distances.cuh"
#include "exact_kernels.cuh"
#include "utils.cuh"

#include "barnes_hut.cuh"
#include "exact_tsne.cuh"

namespace ML {

void TSNE_fit(const raft::handle_t &handle, const float *X, float *Y,
              const int n, const int p, int64_t *knn_indices, float *knn_dists,
              const int dim, int n_neighbors, const float theta,
              const float epssq, float perplexity,
              const int perplexity_max_iter, const float perplexity_tol,
              const float early_exaggeration, const int exaggeration_iter,
              const float min_gain, const float pre_learning_rate,
              const float post_learning_rate, const int max_iter,
              const float min_grad_norm, const float pre_momentum,
              const float post_momentum, const long long random_state,
              int verbosity, const bool initialize_embeddings,
              bool barnes_hut) {
  ASSERT(n > 0 && p > 0 && dim > 0 && n_neighbors > 0 && X != NULL && Y != NULL,
         "Wrong input args");
  ML::Logger::get().setLevel(verbosity);
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

  CUML_LOG_DEBUG("Data size = (%d, %d) with dim = %d perplexity = %f", n, p,
                 dim, perplexity);
  if (perplexity < 5 or perplexity > 50)
    CUML_LOG_WARN(
      "Perplexity should be within ranges (5, 50). Your results might be a"
      " bit strange...");
  if (n_neighbors < perplexity * 3.0f)
    CUML_LOG_WARN(
      "# of Nearest Neighbors should be at least 3 * perplexity. Your results"
      " might be a bit strange...");

  auto d_alloc = handle.get_device_allocator();
  cudaStream_t stream = handle.get_stream();

  START_TIMER;
  //---------------------------------------------------
  // Get distances
  CUML_LOG_DEBUG("Getting distances.");
  MLCommon::device_buffer<int64_t> *knn_indices_b = nullptr;
  MLCommon::device_buffer<float> *knn_dists_b = nullptr;

  if (!knn_indices || !knn_dists) {
    ASSERT(!knn_indices && !knn_dists,
           "Either both or none of the KNN parameters should be provided");
    knn_indices_b =
      new MLCommon::device_buffer<int64_t>(d_alloc, stream, n * n_neighbors);
    knn_dists_b =
      new MLCommon::device_buffer<float>(d_alloc, stream, n * n_neighbors);
    knn_indices = knn_indices_b->data();
    knn_dists = knn_dists_b->data();
    TSNE::get_distances(X, n, p, knn_indices, knn_dists, n_neighbors, d_alloc,
                        stream);
  }
  //---------------------------------------------------
  END_TIMER(DistancesTime);

  START_TIMER;
  //---------------------------------------------------
  // Normalize distances
  CUML_LOG_DEBUG("Now normalizing distances so exp(D) doesn't explode.");
  TSNE::normalize_distances(n, knn_dists, n_neighbors, stream);
  //---------------------------------------------------
  END_TIMER(NormalizeTime);

  START_TIMER;
  //---------------------------------------------------
  // Optimal perplexity
  CUML_LOG_DEBUG("Searching for optimal perplexity via bisection search.");
  MLCommon::device_buffer<float> P(d_alloc, stream, n * n_neighbors);
  TSNE::perplexity_search(knn_dists, P.data(), perplexity, perplexity_max_iter,
                          perplexity_tol, n, n_neighbors, handle);
  //---------------------------------------------------
  END_TIMER(PerplexityTime);

  START_TIMER;
  //---------------------------------------------------
  // Convert data to COO layout
  raft::sparse::COO<float> COO_Matrix(d_alloc, stream);
  TSNE::symmetrize_perplexity(P.data(), knn_indices, n, n_neighbors,
                              early_exaggeration, &COO_Matrix, stream, handle);
  P.release(stream);
  if (knn_indices_b) delete knn_indices_b;
  if (knn_dists_b) delete knn_dists_b;
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
                     min_grad_norm, pre_momentum, post_momentum, random_state,
                     initialize_embeddings);
  } else {
    TSNE::Exact_TSNE(VAL, COL, ROW, NNZ, handle, Y, n, dim, early_exaggeration,
                     exaggeration_iter, min_gain, pre_learning_rate,
                     post_learning_rate, max_iter, min_grad_norm, pre_momentum,
                     post_momentum, random_state, initialize_embeddings);
  }
}

}  // namespace ML
