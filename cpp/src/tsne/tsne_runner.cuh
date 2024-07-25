/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include "barnes_hut_tsne.cuh"
#include "distances.cuh"
#include "exact_kernels.cuh"
#include "exact_tsne.cuh"
#include "fft_tsne.cuh"
#include "utils.cuh"

#include <cuml/common/logger.hpp>
#include <cuml/manifold/common.hpp>

#include <raft/core/handle.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/divide.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/transform.h>

#include <pca/pca.cuh>

namespace ML {

template <class T, template <class> class U>
inline constexpr bool is_instance_of = std::false_type{};

template <template <class> class U, class V>
inline constexpr bool is_instance_of<U<V>, U> = std::true_type{};

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

    auto stream         = handle.get_stream();
    const value_idx dim = params.dim;

    if (params.init == TSNE_INIT::RANDOM) {
      random_vector(Y, -0.0001f, 0.0001f, n * dim, stream, params.random_state);
    } else if (params.init == TSNE_INIT::PCA) {
      auto components          = raft::make_device_matrix<float>(handle, p, dim);
      auto explained_var       = raft::make_device_vector<float>(handle, dim);
      auto explained_var_ratio = raft::make_device_vector<float>(handle, dim);
      auto singular_vals       = raft::make_device_vector<float>(handle, dim);
      auto mu                  = raft::make_device_vector<float>(handle, p);
      auto noise_vars          = raft::make_device_scalar<float>(handle, 0);

      paramsPCA prms;
      prms.n_cols       = p;
      prms.n_rows       = n;
      prms.n_components = dim;
      prms.whiten       = false;
      prms.algorithm    = solver::COV_EIG_DQ;

      if constexpr (!is_instance_of<tsne_input, manifold_dense_inputs_t>) {
        throw std::runtime_error("The tsne_input must be of type manifold_dense_inputs_t");
      } else {
        pcaFitTransform(handle,
                        input.X,
                        Y,
                        components.data_handle(),
                        explained_var.data_handle(),
                        explained_var_ratio.data_handle(),
                        singular_vals.data_handle(),
                        mu.data_handle(),
                        noise_vars.data_handle(),
                        prms,
                        stream);

        auto mean_result       = raft::make_device_vector<float, int>(handle, dim);
        auto stddev_result     = raft::make_device_vector<float, int>(handle, dim);
        const float multiplier = 1e-4;

        auto Y_view = raft::make_device_matrix_view<float, int, raft::col_major>(Y, n, dim);
        auto Y_view_const =
          raft::make_device_matrix_view<const float, int, raft::col_major>(Y, n, dim);

        auto mean_result_view       = mean_result.view();
        auto mean_result_view_const = raft::make_const_mdspan(mean_result.view());

        auto stddev_result_view = stddev_result.view();

        auto h_multiplier_view_const = raft::make_host_scalar_view<const float>(&multiplier);

        raft::stats::mean(handle, Y_view_const, mean_result_view, false);
        raft::stats::stddev(
          handle, Y_view_const, mean_result_view_const, stddev_result_view, false);

        divide_scalar_device(Y_view, Y_view_const, stddev_result_view);
        raft::linalg::multiply_scalar(handle, Y_view_const, Y_view, h_multiplier_view_const);
      }
    }
  }

  void divide_scalar_device(
    raft::device_matrix_view<float, int, raft::col_major>& Y_view,
    raft::device_matrix_view<const float, int, raft::col_major>& Y_view_const,
    raft::device_vector_view<float, int>& stddev_result_view)
  {
    raft::linalg::unary_op(handle,
                           Y_view_const,
                           Y_view,
                           [device_scalar = stddev_result_view.data_handle()] __device__(auto y) {
                             return y / *device_scalar;
                           });
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
