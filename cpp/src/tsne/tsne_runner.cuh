/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include "distances.cuh"
#include "exact_kernels.cuh"
#include "utils.cuh"
#include <cuml/common/logger.hpp>
#include <cuml/manifold/common.hpp>
#include <raft/core/cudart_utils.hpp>
#include <raft/distance/distance_types.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/transform.h>

#include "barnes_hut_tsne.cuh"
#include "exact_tsne.cuh"
#include "fft_tsne.cuh"

namespace ML {

template <typename tsne_input, typename value_idx, typename value_t>
class TSNE_runner {
 public:
  TSNE_runner(const raft::handle_t& handle_,
              tsne_input& input_,
              knn_graph<value_idx, value_t>& k_graph_,
              TSNEParams& params_)
    : handle(handle_),
      input(input_),
      k_graph(k_graph_),
      params(params_),
      COO_Matrix(handle_.get_stream())
  {
    this->n = input.n;
    this->p = input.d;
    this->Y = input.y;

    ML::Logger::get().setLevel(params.verbosity);
    if (params.dim > 2 and params.algorithm != TSNE_ALGORITHM::EXACT) {
      params.algorithm = TSNE_ALGORITHM::EXACT;
      CUML_LOG_WARN(
        "Barnes Hut and FFT only work for dim == 2. Switching to exact "
        "solution.");
    }
    if (params.n_neighbors > n) params.n_neighbors = n;
    if (params.n_neighbors > 1023) {
      CUML_LOG_WARN("FAISS only supports maximum n_neighbors = 1023.");
      params.n_neighbors = 1023;
    }
    // Perplexity must be less than number of datapoints
    // "How to Use t-SNE Effectively" https://distill.pub/2016/misread-tsne/
    if (params.perplexity > n) params.perplexity = n;

    CUML_LOG_DEBUG(
      "Data size = (%d, %d) with dim = %d perplexity = %f", n, p, params.dim, params.perplexity);
    if (params.perplexity < 5 or params.perplexity > 50)
      CUML_LOG_WARN(
        "Perplexity should be within ranges (5, 50). Your results might be a"
        " bit strange...");
    if (params.n_neighbors < params.perplexity * 3.0f)
      CUML_LOG_WARN(
        "# of Nearest Neighbors should be at least 3 * perplexity. Your results"
        " might be a bit strange...");
  }

  value_t run()
  {
    distance_and_perplexity();

    const auto NNZ  = COO_Matrix.nnz;
    auto* VAL       = COO_Matrix.vals();
    const auto* COL = COO_Matrix.cols();
    const auto* ROW = COO_Matrix.rows();
    //---------------------------------------------------

    switch (params.algorithm) {
      case TSNE_ALGORITHM::BARNES_HUT:
        return TSNE::Barnes_Hut(VAL, COL, ROW, NNZ, handle, Y, n, params);
      case TSNE_ALGORITHM::FFT: return TSNE::FFT_TSNE(VAL, COL, ROW, NNZ, handle, Y, n, params);
      case TSNE_ALGORITHM::EXACT: return TSNE::Exact_TSNE(VAL, COL, ROW, NNZ, handle, Y, n, params);
    }
    return 0;
  }

 private:
  void distance_and_perplexity()
  {
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

      indices   = rmm::device_uvector<value_idx>(n * params.n_neighbors, stream);
      distances = rmm::device_uvector<value_t>(n * params.n_neighbors, stream);

      k_graph.knn_indices = indices.data();
      k_graph.knn_dists   = distances.data();

      TSNE::get_distances(handle, input, k_graph, stream, params.metric, params.p);
    }

    if (params.square_distances) {
      auto policy = handle.get_thrust_policy();

      thrust::transform(policy,
                        k_graph.knn_dists,
                        k_graph.knn_dists + n * params.n_neighbors,
                        k_graph.knn_dists,
                        TSNE::FunctionalSquare());
    }

    //---------------------------------------------------
    END_TIMER(DistancesTime);

    START_TIMER;
    //---------------------------------------------------
    // Normalize distances
    CUML_LOG_DEBUG("Now normalizing distances so exp(D) doesn't explode.");
    TSNE::normalize_distances(k_graph.knn_dists, n * params.n_neighbors, stream);
    //---------------------------------------------------
    END_TIMER(NormalizeTime);

    START_TIMER;
    //---------------------------------------------------
    // Optimal perplexity
    CUML_LOG_DEBUG("Searching for optimal perplexity via bisection search.");
    rmm::device_uvector<value_t> P(n * params.n_neighbors, stream);
    TSNE::perplexity_search(k_graph.knn_dists,
                            P.data(),
                            params.perplexity,
                            params.perplexity_max_iter,
                            params.perplexity_tol,
                            n,
                            params.n_neighbors,
                            handle);

    //---------------------------------------------------
    END_TIMER(PerplexityTime);

    START_TIMER;
    //---------------------------------------------------
    // Normalize perplexity to prepare for symmetrization
    raft::linalg::scalarMultiply(P.data(), P.data(), 1.0f / (2.0f * n), P.size(), stream);
    //---------------------------------------------------
    END_TIMER(NormalizeTime);

    START_TIMER;
    //---------------------------------------------------
    // Convert data to COO layout
    TSNE::symmetrize_perplexity(P.data(),
                                k_graph.knn_indices,
                                n,
                                params.n_neighbors,
                                params.early_exaggeration,
                                &COO_Matrix,
                                stream,
                                handle);
    END_TIMER(SymmetrizeTime);
  }

 public:
  raft::sparse::COO<value_t, value_idx> COO_Matrix;

 private:
  const raft::handle_t& handle;
  tsne_input& input;
  knn_graph<value_idx, value_t>& k_graph;
  TSNEParams& params;

  value_idx n, p;
  value_t* Y;
};

}  // namespace ML
