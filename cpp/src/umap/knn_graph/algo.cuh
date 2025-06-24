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

#include <cuml/common/utils.hpp>
#include <cuml/manifold/common.hpp>
#include <cuml/manifold/umapparams.h>
#include <cuml/neighbors/knn_sparse.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <stdint.h>

#include <iostream>

namespace UMAPAlgo {
namespace kNNGraph {
namespace Algo {

/**
 * Initial implementation calls out to FAISS to do its work.
 */

template <typename value_idx = int64_t, typename value_t = float, typename umap_inputs>
void launcher(const raft::handle_t& handle,
              const umap_inputs& inputsA,
              const umap_inputs& inputsB,
              ML::knn_graph<value_idx, value_t>& out,
              int n_neighbors,
              const ML::UMAPParams* params,
              cudaStream_t stream);

// Instantiation for dense inputs, int64_t indices
template <>
inline void launcher(const raft::handle_t& handle,
                     const ML::manifold_dense_inputs_t<float>& inputsA,
                     const ML::manifold_dense_inputs_t<float>& inputsB,
                     ML::knn_graph<int64_t, float>& out,
                     int n_neighbors,
                     const ML::UMAPParams* params,
                     cudaStream_t stream)
{
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, inputsA.X));
  bool data_on_device = attr.type == cudaMemoryTypeDevice;

  if (params->build_algo == ML::UMAPParams::graph_build_algo::BRUTE_FORCE_KNN) {
    auto idx = [&]() {
      if (data_on_device) {  // inputsA on device
        return cuvs::neighbors::brute_force::build(
          handle,
          {static_cast<cuvs::distance::DistanceType>(params->metric), params->p},
          raft::make_device_matrix_view<const float, int64_t>(inputsA.X, inputsA.n, inputsA.d));
      } else {  // inputsA on host
        return cuvs::neighbors::brute_force::build(
          handle,
          {static_cast<cuvs::distance::DistanceType>(params->metric), params->p},
          raft::make_host_matrix_view<const float, int64_t>(inputsA.X, inputsA.n, inputsA.d));
      }
    }();
    cuvs::neighbors::brute_force::search(
      handle,
      idx,
      raft::make_device_matrix_view<const float, int64_t>(inputsB.X, inputsB.n, inputsB.d),
      raft::make_device_matrix_view<int64_t, int64_t>(out.knn_indices, inputsB.n, n_neighbors),
      raft::make_device_matrix_view<float, int64_t>(out.knn_dists, inputsB.n, n_neighbors));
  } else {  // nn_descent
    RAFT_EXPECTS(
      static_cast<size_t>(n_neighbors) <= params->build_params.nn_descent_params.graph_degree,
      "n_neighbors should be smaller than the graph degree computed by nn descent");
    RAFT_EXPECTS(
      params->build_params.nn_descent_params.graph_degree <=
        params->build_params.nn_descent_params.intermediate_graph_degree,
      "graph_degree should be smaller than intermediate_graph_degree computed by nn descent");

    auto all_neighbors_params           = cuvs::neighbors::all_neighbors::all_neighbors_params{};
    all_neighbors_params.overlap_factor = params->build_params.overlap_factor;
    all_neighbors_params.n_clusters     = params->build_params.n_clusters;
    all_neighbors_params.metric         = static_cast<cuvs::distance::DistanceType>(params->metric);

    auto nn_descent_params =
      cuvs::neighbors::all_neighbors::graph_build_params::nn_descent_params{};
    nn_descent_params.graph_degree = params->build_params.nn_descent_params.graph_degree;
    nn_descent_params.intermediate_graph_degree =
      params->build_params.nn_descent_params.intermediate_graph_degree;
    nn_descent_params.max_iterations = params->build_params.nn_descent_params.max_iterations;
    nn_descent_params.termination_threshold =
      params->build_params.nn_descent_params.termination_threshold;
    nn_descent_params.metric = static_cast<cuvs::distance::DistanceType>(params->metric);
    all_neighbors_params.graph_build_params = nn_descent_params;

    auto indices_view =
      raft::make_device_matrix_view<int64_t, int64_t>(out.knn_indices, inputsB.n, n_neighbors);
    auto distances_view =
      raft::make_device_matrix_view<float, int64_t>(out.knn_dists, inputsB.n, n_neighbors);

    if (data_on_device) {  // inputsA on device
      cuvs::neighbors::all_neighbors::build(
        handle,
        all_neighbors_params,
        raft::make_device_matrix_view<const float, int64_t>(inputsA.X, inputsA.n, inputsA.d),
        indices_view,
        distances_view);
    } else {  // inputsA on host
      cuvs::neighbors::all_neighbors::build(
        handle,
        all_neighbors_params,
        raft::make_host_matrix_view<const float, int64_t>(inputsA.X, inputsA.n, inputsA.d),
        indices_view,
        distances_view);
    }
  }
}

// Instantiation for dense inputs, int indices
template <>
inline void launcher(const raft::handle_t& handle,
                     const ML::manifold_dense_inputs_t<float>& inputsA,
                     const ML::manifold_dense_inputs_t<float>& inputsB,
                     ML::knn_graph<int, float>& out,
                     int n_neighbors,
                     const ML::UMAPParams* params,
                     cudaStream_t stream)
{
  throw raft::exception("Dense KNN doesn't yet support 32-bit integer indices");
}

template <>
inline void launcher(const raft::handle_t& handle,
                     const ML::manifold_sparse_inputs_t<int, float>& inputsA,
                     const ML::manifold_sparse_inputs_t<int, float>& inputsB,
                     ML::knn_graph<int, float>& out,
                     int n_neighbors,
                     const ML::UMAPParams* params,
                     cudaStream_t stream)
{
  RAFT_EXPECTS(params->build_algo == ML::UMAPParams::graph_build_algo::BRUTE_FORCE_KNN,
               "nn_descent does not support sparse inputs");

  auto a_structure = raft::make_device_compressed_structure_view<int, int, int>(
    inputsA.indptr, inputsA.indices, inputsA.n, inputsA.d, inputsA.nnz);
  auto a_csr = raft::make_device_csr_matrix_view<const float>(inputsA.data, a_structure);

  auto b_structure = raft::make_device_compressed_structure_view<int, int, int>(
    inputsB.indptr, inputsB.indices, inputsB.n, inputsB.d, inputsB.nnz);
  auto b_csr = raft::make_device_csr_matrix_view<const float>(inputsB.data, b_structure);

  cuvs::neighbors::brute_force::sparse_search_params search_params;
  search_params.batch_size_index = ML::Sparse::DEFAULT_BATCH_SIZE;
  search_params.batch_size_query = ML::Sparse::DEFAULT_BATCH_SIZE;

  auto index = cuvs::neighbors::brute_force::build(
    handle, a_csr, static_cast<cuvs::distance::DistanceType>(params->metric), params->p);

  cuvs::neighbors::brute_force::search(
    handle,
    search_params,
    index,
    b_csr,
    raft::make_device_matrix_view<int, int64_t>(out.knn_indices, inputsB.n, n_neighbors),
    raft::make_device_matrix_view<float, int64_t>(out.knn_dists, inputsB.n, n_neighbors));
}

template <>
inline void launcher(const raft::handle_t& handle,
                     const ML::manifold_sparse_inputs_t<int64_t, float>& inputsA,
                     const ML::manifold_sparse_inputs_t<int64_t, float>& inputsB,
                     ML::knn_graph<int64_t, float>& out,
                     int n_neighbors,
                     const ML::UMAPParams* params,
                     cudaStream_t stream)
{
  throw raft::exception("Sparse KNN doesn't support 64-bit integer indices");
}

template <>
inline void launcher(const raft::handle_t& handle,
                     const ML::manifold_precomputed_knn_inputs_t<int64_t, float>& inputsA,
                     const ML::manifold_precomputed_knn_inputs_t<int64_t, float>& inputsB,
                     ML::knn_graph<int64_t, float>& out,
                     int n_neighbors,
                     const ML::UMAPParams* params,
                     cudaStream_t stream)
{
  out.knn_indices = inputsA.knn_graph.knn_indices;
  out.knn_dists   = inputsA.knn_graph.knn_dists;
}

// Instantiation for precomputed inputs, int indices
template <>
inline void launcher(const raft::handle_t& handle,
                     const ML::manifold_precomputed_knn_inputs_t<int, float>& inputsA,
                     const ML::manifold_precomputed_knn_inputs_t<int, float>& inputsB,
                     ML::knn_graph<int, float>& out,
                     int n_neighbors,
                     const ML::UMAPParams* params,
                     cudaStream_t stream)
{
  out.knn_indices = inputsA.knn_graph.knn_indices;
  out.knn_dists   = inputsA.knn_graph.knn_dists;
}

}  // namespace Algo
}  // namespace kNNGraph
};  // namespace UMAPAlgo
