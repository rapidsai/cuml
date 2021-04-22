/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <raft/cudart_utils.h>
#include <cuml/common/logger.hpp>
#include <cuml/manifold/common.hpp>
#include <rmm/device_uvector.hpp>
#include "distances.cuh"
#include "exact_kernels.cuh"
#include "utils.cuh"

#include "barnes_hut_tsne.cuh"
#include "exact_tsne.cuh"
#include "fft_tsne.cuh"

namespace ML {

template <typename tsne_input, typename value_idx, typename value_t>
class TSNE_runner {
 public:
  TSNE_runner(const raft::handle_t &handle_, tsne_input &input_,
              knn_graph<value_idx, value_t> &k_graph_, const value_idx dim_,
              const float theta_, const float epssq_, float perplexity_,
              const int perplexity_max_iter_, const float perplexity_tol_,
              const float early_exaggeration_, const float late_exaggeration_,
              const int exaggeration_iter_, const float min_gain_,
              const float pre_learning_rate_, const float post_learning_rate_,
              const int max_iter_, const float min_grad_norm_,
              const float pre_momentum_, const float post_momentum_,
              const long long random_state_, int verbosity_,
              const bool initialize_embeddings_, const bool square_distances_,
              TSNE_ALGORITHM algorithm_)
    : handle(handle_),
      input(input_),
      k_graph(k_graph_),
      dim(dim_),
      theta(theta_),
      epssq(epssq_),
      perplexity(perplexity_),
      perplexity_max_iter(perplexity_max_iter_),
      perplexity_tol(perplexity_tol_),
      early_exaggeration(early_exaggeration_),
      late_exaggeration(late_exaggeration_),
      exaggeration_iter(exaggeration_iter_),
      min_gain(min_gain_),
      pre_learning_rate(pre_learning_rate_),
      post_learning_rate(post_learning_rate_),
      max_iter(max_iter_),
      min_grad_norm(min_grad_norm_),
      pre_momentum(pre_momentum_),
      post_momentum(post_momentum_),
      random_state(random_state_),
      verbosity(verbosity_),
      initialize_embeddings(initialize_embeddings_),
      square_distances(square_distances_),
      algorithm(algorithm_),
      COO_Matrix(handle_.get_device_allocator(), handle_.get_stream()) {
    this->n = input.n;
    this->p = input.d;
    this->Y = input.y;
    this->n_neighbors = k_graph.n_neighbors;

    ML::Logger::get().setLevel(verbosity);
    if (dim > 2 and algorithm != TSNE_ALGORITHM::EXACT) {
      algorithm = TSNE_ALGORITHM::EXACT;
      CUML_LOG_WARN(
        "Barnes Hut and FFT only work for dim == 2. Switching to exact "
        "solution.");
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
  }

  void run() {
    distance_and_perplexity();

    const auto NNZ = COO_Matrix.nnz;
    auto *VAL = COO_Matrix.vals();
    const auto *COL = COO_Matrix.cols();
    const auto *ROW = COO_Matrix.rows();
    //---------------------------------------------------

    switch (algorithm) {
      case TSNE_ALGORITHM::BARNES_HUT:
        TSNE::Barnes_Hut(VAL, COL, ROW, NNZ, handle, Y, n, theta, epssq,
                         early_exaggeration, exaggeration_iter, min_gain,
                         pre_learning_rate, post_learning_rate, max_iter,
                         min_grad_norm, pre_momentum, post_momentum,
                         random_state, initialize_embeddings);
        break;
      case TSNE_ALGORITHM::FFT:
        TSNE::FFT_TSNE(VAL, COL, ROW, NNZ, handle, Y, n, early_exaggeration,
                       late_exaggeration, exaggeration_iter, pre_learning_rate,
                       post_learning_rate, max_iter, min_grad_norm,
                       pre_momentum, post_momentum, random_state,
                       initialize_embeddings);
        break;
      case TSNE_ALGORITHM::EXACT:
        TSNE::Exact_TSNE(VAL, COL, ROW, NNZ, handle, Y, n, dim,
                         early_exaggeration, exaggeration_iter, min_gain,
                         pre_learning_rate, post_learning_rate, max_iter,
                         min_grad_norm, pre_momentum, post_momentum,
                         random_state, initialize_embeddings);
        break;
    }
  }

 private:
  void distance_and_perplexity() {
    START_TIMER;

    //---------------------------------------------------
    // Get distances
    CUML_LOG_DEBUG("Getting distances.");

    auto stream = handle.get_stream();

    rmm::device_uvector<value_idx> indices(0, stream);
    rmm::device_uvector<value_t> distances(0, stream);

    if (!k_graph.knn_indices || !k_graph.knn_dists) {
      ASSERT(!k_graph.knn_indices && !k_graph.knn_dists,
             "Either both or none of the KNN parameters should be provided");

      indices = rmm::device_uvector<value_idx>(n * n_neighbors, stream);
      distances = rmm::device_uvector<value_t>(n * n_neighbors, stream);

      k_graph.knn_indices = indices.data();
      k_graph.knn_dists = distances.data();

      TSNE::get_distances(handle, input, k_graph, stream);
    }

    if (square_distances) {
      auto policy = rmm::exec_policy(stream);

      thrust::transform(policy, k_graph.knn_dists,
                        k_graph.knn_dists + n * n_neighbors, k_graph.knn_dists,
                        TSNE::FunctionalSquare());
    }

    //---------------------------------------------------
    END_TIMER(DistancesTime);

    START_TIMER;
    //---------------------------------------------------
    // Normalize distances
    CUML_LOG_DEBUG("Now normalizing distances so exp(D) doesn't explode.");
    TSNE::normalize_distances(n, k_graph.knn_dists, n_neighbors, stream);
    //---------------------------------------------------
    END_TIMER(NormalizeTime);

    START_TIMER;
    //---------------------------------------------------
    // Optimal perplexity
    CUML_LOG_DEBUG("Searching for optimal perplexity via bisection search.");
    rmm::device_uvector<value_t> P(n * n_neighbors, stream);
    TSNE::perplexity_search(k_graph.knn_dists, P.data(), perplexity,
                            perplexity_max_iter, perplexity_tol, n, n_neighbors,
                            handle);

    //---------------------------------------------------
    END_TIMER(PerplexityTime);

    START_TIMER;
    //---------------------------------------------------
    // Convert data to COO layout
    TSNE::symmetrize_perplexity(P.data(), k_graph.knn_indices, n, n_neighbors,
                                early_exaggeration, &COO_Matrix, stream,
                                handle);
    END_TIMER(SymmetrizeTime);
  }

  const raft::handle_t &handle;
  tsne_input &input;
  knn_graph<value_idx, value_t> &k_graph;
  const value_idx dim;
  int n_neighbors;
  const float theta;
  const float epssq;
  float perplexity;
  const int perplexity_max_iter;
  const float perplexity_tol;
  const float early_exaggeration;
  const float late_exaggeration;
  const int exaggeration_iter;
  const float min_gain;
  const float pre_learning_rate;
  const float post_learning_rate;
  const int max_iter;
  const float min_grad_norm;
  const float pre_momentum;
  const float post_momentum;
  const long long random_state;
  int verbosity;
  const bool initialize_embeddings;
  const bool square_distances;
  TSNE_ALGORITHM algorithm;

  raft::sparse::COO<value_t, value_idx> COO_Matrix;
  value_idx n, p;
  value_t *Y;
};

}  // namespace ML
