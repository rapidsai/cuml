/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "runner.cuh"

#include <cuml/manifold/common.hpp>
#include <cuml/manifold/umap.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/core/handle.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/util/cuda_utils.cuh>

#include <stdint.h>

#include <iostream>

namespace ML {
namespace UMAP {

static const int TPB_X = 256;

inline bool dispatch_to_uint64_t(int n_rows, int n_neighbors, int n_components)
{
  // The fuzzy simplicial set graph can have at most 2 * n * n_neighbors elements after
  // symmetrization and removal of zeroes
  uint64_t nnz1 = 2 * static_cast<uint64_t>(n_rows) * n_neighbors;
  // The embeddings have n * n_neighbors elements
  uint64_t nnz2 = static_cast<uint64_t>(n_rows) * n_components;
  return nnz1 > std::numeric_limits<int32_t>::max() || nnz2 > std::numeric_limits<int32_t>::max();
}

template <typename nnz_t>
inline std::unique_ptr<raft::sparse::COO<float, int>> _get_graph(
  const raft::handle_t& handle,
  float* X,  // input matrix
  float* y,  // labels
  int n,
  int d,
  knn_indices_dense_t* knn_indices,  // precomputed indices
  float* knn_dists,                  // precomputed distances
  UMAPParams* params)
{
  auto graph = std::make_unique<raft::sparse::COO<float>>(handle.get_stream());
  if (knn_indices != nullptr && knn_dists != nullptr) {
    CUML_LOG_DEBUG("Calling UMAP::get_graph() with precomputed KNN");

    manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float> inputs(
      knn_indices, knn_dists, y, n, d, params->n_neighbors);
    if (y != nullptr) {
      UMAPAlgo::_get_graph_supervised<knn_indices_dense_t,
                                      float,
                                      manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float>,
                                      nnz_t,
                                      TPB_X>(handle, inputs, params, graph.get());
    } else {
      UMAPAlgo::_get_graph<knn_indices_dense_t,
                           float,
                           manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float>,
                           nnz_t,
                           TPB_X>(handle, inputs, params, graph.get());
    }
    return graph;
  } else {
    manifold_dense_inputs_t<float> inputs(X, y, n, d);
    if (y != nullptr) {
      UMAPAlgo::_get_graph_supervised<knn_indices_dense_t,
                                      float,
                                      manifold_dense_inputs_t<float>,
                                      nnz_t,
                                      TPB_X>(handle, inputs, params, graph.get());
    } else {
      UMAPAlgo::
        _get_graph<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
          handle, inputs, params, graph.get());
    }
    return graph;
  }
}

template <typename nnz_t>
inline void _refine(const raft::handle_t& handle,
                    float* X,
                    int n,
                    int d,
                    raft::sparse::COO<float>* graph,
                    UMAPParams* params,
                    float* embeddings)
{
  CUML_LOG_DEBUG("Calling UMAP::refine() with precomputed KNN");
  manifold_dense_inputs_t<float> inputs(X, nullptr, n, d);
  UMAPAlgo::_refine<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
    handle, inputs, params, graph, embeddings);
}

template <typename nnz_t>
inline void _init_and_refine(const raft::handle_t& handle,
                             float* X,
                             int n,
                             int d,
                             raft::sparse::COO<float>* graph,
                             UMAPParams* params,
                             float* embeddings)
{
  CUML_LOG_DEBUG("Calling UMAP::init_and_refine() with precomputed KNN");
  manifold_dense_inputs_t<float> inputs(X, nullptr, n, d);
  UMAPAlgo::
    _init_and_refine<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
      handle, inputs, params, graph, embeddings);
}

template <typename nnz_t>
inline void _fit(const raft::handle_t& handle,
                 float* X,
                 float* y,
                 int n,
                 int d,
                 knn_indices_dense_t* knn_indices,
                 float* knn_dists,
                 UMAPParams* params,
                 std::unique_ptr<rmm::device_buffer>& embeddings,
                 raft::host_coo_matrix<float, int, int, uint64_t>& graph)
{
  if (knn_indices != nullptr && knn_dists != nullptr) {
    CUML_LOG_DEBUG("Calling UMAP::fit() with precomputed KNN");

    manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float> inputs(
      knn_indices, knn_dists, y, n, d, params->n_neighbors);
    if (y != nullptr) {
      UMAPAlgo::_fit_supervised<knn_indices_dense_t,
                                float,
                                manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float>,
                                nnz_t,
                                TPB_X>(handle, inputs, params, embeddings, graph);
    } else {
      UMAPAlgo::_fit<knn_indices_dense_t,
                     float,
                     manifold_precomputed_knn_inputs_t<knn_indices_dense_t, float>,
                     nnz_t,
                     TPB_X>(handle, inputs, params, embeddings, graph);
    }

  } else {
    manifold_dense_inputs_t<float> inputs(X, y, n, d);
    if (y != nullptr) {
      UMAPAlgo::
        _fit_supervised<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
          handle, inputs, params, embeddings, graph);
    } else {
      UMAPAlgo::_fit<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
        handle, inputs, params, embeddings, graph);
    }
  }
}

template <typename nnz_t>
inline void _fit_sparse(const raft::handle_t& handle,
                        int* indptr,
                        int* indices,
                        float* data,
                        size_t nnz,
                        float* y,
                        int n,
                        int d,
                        int* knn_indices,
                        float* knn_dists,
                        UMAPParams* params,
                        std::unique_ptr<rmm::device_buffer>& embeddings,
                        raft::host_coo_matrix<float, int, int, uint64_t>& graph)
{
  if (knn_indices != nullptr && knn_dists != nullptr) {
    CUML_LOG_DEBUG("Calling UMAP::fit_sparse() with precomputed KNN");

    manifold_precomputed_knn_inputs_t<knn_indices_sparse_t, float> inputs(
      knn_indices, knn_dists, y, n, d, params->n_neighbors);
    if (y != nullptr) {
      UMAPAlgo::_fit_supervised<knn_indices_sparse_t,
                                float,
                                manifold_precomputed_knn_inputs_t<knn_indices_sparse_t, float>,
                                nnz_t,
                                TPB_X>(handle, inputs, params, embeddings, graph);
    } else {
      UMAPAlgo::_fit<knn_indices_sparse_t,
                     float,
                     manifold_precomputed_knn_inputs_t<knn_indices_sparse_t, float>,
                     nnz_t,
                     TPB_X>(handle, inputs, params, embeddings, graph);
    }
  } else {
    manifold_sparse_inputs_t<int, float> inputs(indptr, indices, data, y, nnz, n, d);
    if (y != nullptr) {
      UMAPAlgo::_fit_supervised<knn_indices_sparse_t,
                                float,
                                manifold_sparse_inputs_t<knn_indices_sparse_t, float>,
                                nnz_t,
                                TPB_X>(handle, inputs, params, embeddings, graph);
    } else {
      UMAPAlgo::_fit<knn_indices_sparse_t,
                     float,
                     manifold_sparse_inputs_t<knn_indices_sparse_t, float>,
                     nnz_t,
                     TPB_X>(handle, inputs, params, embeddings, graph);
    }
  }
}

template <typename nnz_t>
inline void _transform(const raft::handle_t& handle,
                       float* X,
                       int n,
                       int d,
                       float* orig_X,
                       int orig_n,
                       float* embedding,
                       int embedding_n,
                       UMAPParams* params,
                       float* transformed)
{
  RAFT_EXPECTS(params->build_algo == ML::UMAPParams::graph_build_algo::BRUTE_FORCE_KNN,
               "build algo nn_descent not supported for transform()");
  manifold_dense_inputs_t<float> inputs(X, nullptr, n, d);
  manifold_dense_inputs_t<float> orig_inputs(orig_X, nullptr, orig_n, d);
  UMAPAlgo::_transform<knn_indices_dense_t, float, manifold_dense_inputs_t<float>, nnz_t, TPB_X>(
    handle, inputs, orig_inputs, embedding, embedding_n, params, transformed);
}

template <typename nnz_t>
inline void _transform_sparse(const raft::handle_t& handle,
                              int* indptr,
                              int* indices,
                              float* data,
                              size_t nnz,
                              int n,
                              int d,
                              int* orig_x_indptr,
                              int* orig_x_indices,
                              float* orig_x_data,
                              size_t orig_nnz,
                              int orig_n,
                              float* embedding,
                              int embedding_n,
                              UMAPParams* params,
                              float* transformed)
{
  RAFT_EXPECTS(params->build_algo == ML::UMAPParams::graph_build_algo::BRUTE_FORCE_KNN,
               "build algo nn_descent not supported for transform()");
  manifold_sparse_inputs_t<knn_indices_sparse_t, float> inputs(
    indptr, indices, data, nullptr, nnz, n, d);
  manifold_sparse_inputs_t<knn_indices_sparse_t, float> orig_x_inputs(
    orig_x_indptr, orig_x_indices, orig_x_data, nullptr, orig_nnz, orig_n, d);

  UMAPAlgo::
    _transform<knn_indices_sparse_t, float, manifold_sparse_inputs_t<int, float>, nnz_t, TPB_X>(
      handle, inputs, orig_x_inputs, embedding, embedding_n, params, transformed);
}

}  // namespace UMAP
}  // namespace ML
