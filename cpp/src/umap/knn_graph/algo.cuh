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
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
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

auto get_graph_nnd(const raft::handle_t& handle,
                   const ML::manifold_dense_inputs_t<float>& inputs,
                   const ML::UMAPParams* params)
{
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, inputs.X));
  float* ptr = reinterpret_cast<float*>(attr.devicePointer);
  if (ptr != nullptr) {
    auto dataset =
      raft::make_device_matrix_view<const float, int64_t>(inputs.X, inputs.n, inputs.d);
    return cuvs::neighbors::nn_descent::build(handle, params->nn_descent_params, dataset);
  } else {
    auto dataset = raft::make_host_matrix_view<const float, int64_t>(inputs.X, inputs.n, inputs.d);
    return cuvs::neighbors::nn_descent::build(handle, params->nn_descent_params, dataset);
  }
}

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
  if (params->build_algo == ML::UMAPParams::graph_build_algo::BRUTE_FORCE_KNN) {
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, inputsA.X));
    float* ptr = reinterpret_cast<float*>(attr.devicePointer);
    auto idx   = [&]() {
      if (ptr != nullptr) {  // inputsA on device
        return cuvs::neighbors::brute_force::build(
          handle,
          {params->metric, params->p},
          raft::make_device_matrix_view<const float, int64_t>(inputsA.X, inputsA.n, inputsA.d));
      } else {  // inputsA on host
        return cuvs::neighbors::brute_force::build(
          handle,
          {params->metric, params->p},
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
    // TODO:  use nndescent from cuvs
    RAFT_EXPECTS(static_cast<size_t>(n_neighbors) <= params->nn_descent_params.graph_degree,
                 "n_neighbors should be smaller than the graph degree computed by nn descent");
    RAFT_EXPECTS(params->nn_descent_params.return_distances,
                 "return_distances for nn descent should be set to true to be used for UMAP");

    auto graph = get_graph_nnd(handle, inputsA, params);

    // `graph.graph()` is a host array (n x graph_degree).
    // Slice and copy to a temporary host array (n x n_neighbors), then copy
    // that to the output device array `out.knn_indices` (n x n_neighbors).
    // TODO: force graph_degree = n_neighbors so the temporary host array and
    // slice isn't necessary.
    auto temp_indices_h = raft::make_host_matrix<int64_t, int64_t>(inputsA.n, n_neighbors);
    size_t graph_degree = params->nn_descent_params.graph_degree;
#pragma omp parallel for
    for (size_t i = 0; i < static_cast<size_t>(inputsA.n); i++) {
      for (int j = 0; j < n_neighbors; j++) {
        auto target                 = temp_indices_h.data_handle();
        auto source                 = graph.graph().data_handle();
        target[i * n_neighbors + j] = source[i * graph_degree + j];
      }
    }

    raft::copy(handle,
               raft::make_device_matrix_view(out.knn_indices, inputsA.n, n_neighbors),
               temp_indices_h.view());

    // `graph.distances()` is a device array (n x graph_degree).
    // Slice and copy to the output device array `out.knn_dists` (n x n_neighbors).
    // TODO: force graph_degree = n_neighbors so this slice isn't necessary.
    raft::matrix::slice_coordinates coords{static_cast<int64_t>(0),
                                           static_cast<int64_t>(0),
                                           static_cast<int64_t>(inputsA.n),
                                           static_cast<int64_t>(n_neighbors)};
    raft::matrix::slice<float, int64_t, raft::row_major>(
      handle,
      raft::make_const_mdspan(graph.distances().value()),
      raft::make_device_matrix_view(out.knn_dists, inputsA.n, n_neighbors),
      coords);
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

  auto index = cuvs::neighbors::brute_force::build(handle, a_csr, params->metric, params->p);

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
