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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/neighbors/nn_descent.cuh>
#include <raft/neighbors/nn_descent_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>

#include <iostream>

namespace NNDescent = raft::neighbors::experimental::nn_descent;

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

//  Functor to post-process distances as L2Sqrt*
template <typename value_idx, typename value_t = float>
struct DistancePostProcessSqrt : NNDescent::DistEpilogue<value_idx, value_t> {
  DI value_t operator()(value_t value, value_idx row, value_idx col) const { return sqrtf(value); }
};

auto get_graph_nnd(const raft::handle_t& handle,
                   const ML::manifold_dense_inputs_t<float>& inputs,
                   const ML::UMAPParams* params)
{
  auto epilogue = DistancePostProcessSqrt<int64_t, float>{};
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, inputs.X));
  float* ptr = reinterpret_cast<float*>(attr.devicePointer);
  if (ptr != nullptr) {
    auto dataset =
      raft::make_device_matrix_view<const float, int64_t>(inputs.X, inputs.n, inputs.d);
    return NNDescent::build<float, int64_t>(handle, params->nn_descent_params, dataset, epilogue);
  } else {
    auto dataset = raft::make_host_matrix_view<const float, int64_t>(inputs.X, inputs.n, inputs.d);
    return NNDescent::build<float, int64_t>(handle, params->nn_descent_params, dataset, epilogue);
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
    auto idx = cuvs::neighbors::brute_force::build(
      handle,
      raft::make_device_matrix_view<const float, int64_t>(inputsA.X, inputsA.n, inputsA.d),
      params->metric,
      params->p);

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

    auto graph = get_graph_nnd(handle, inputsA, params);

    auto indices_d = raft::make_device_matrix<int64_t, int64_t>(
      handle, inputsA.n, params->nn_descent_params.graph_degree);

    raft::copy(indices_d.data_handle(),
               graph.graph().data_handle(),
               inputsA.n * params->nn_descent_params.graph_degree,
               stream);

    raft::matrix::slice_coordinates coords{static_cast<int64_t>(0),
                                           static_cast<int64_t>(0),
                                           static_cast<int64_t>(inputsA.n),
                                           static_cast<int64_t>(n_neighbors)};

    RAFT_EXPECTS(graph.distances().has_value(),
                 "return_distances for nn descent should be set to true to be used for UMAP");
    auto out_knn_dists_view =
      raft::make_device_matrix_view(out.knn_dists, inputsA.n, (uint64_t)n_neighbors);
    raft::matrix::slice<float, int64_t, raft::row_major>(
      handle, raft::make_const_mdspan(graph.distances().value()), out_knn_dists_view, coords);
    auto out_knn_indices_view =
      raft::make_device_matrix_view(out.knn_indices, inputsA.n, (uint64_t)n_neighbors);
    raft::matrix::slice<int64_t, int64_t, raft::row_major>(
      handle, raft::make_const_mdspan(indices_d.view()), out_knn_indices_view, coords);
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
