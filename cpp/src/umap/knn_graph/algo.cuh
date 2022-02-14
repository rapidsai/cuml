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

#include <cuml/manifold/common.hpp>
#include <cuml/manifold/umapparams.h>
#include <cuml/neighbors/knn_sparse.hpp>
#include <iostream>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/unary_op.hpp>
#include <raft/sparse/selection/knn.hpp>
#include <selection/knn.cuh>

#include <raft/cudart_utils.h>

#include <raft/error.hpp>

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
void launcher(const raft::handle_t& handle,
              const ML::manifold_dense_inputs_t<float>& inputsA,
              const ML::manifold_dense_inputs_t<float>& inputsB,
              ML::knn_graph<int64_t, float>& out,
              int n_neighbors,
              const ML::UMAPParams* params,
              cudaStream_t stream)
{
  std::vector<float*> ptrs(1);
  std::vector<int> sizes(1);
  ptrs[0]  = inputsA.X;
  sizes[0] = inputsA.n;

  raft::spatial::knn::brute_force_knn(handle,
                                      ptrs,
                                      sizes,
                                      inputsA.d,
                                      inputsB.X,
                                      inputsB.n,
                                      out.knn_indices,
                                      out.knn_dists,
                                      n_neighbors);
}

// Instantiation for dense inputs, int indices
template <>
void launcher(const raft::handle_t& handle,
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
void launcher(const raft::handle_t& handle,
              const ML::manifold_sparse_inputs_t<int, float>& inputsA,
              const ML::manifold_sparse_inputs_t<int, float>& inputsB,
              ML::knn_graph<int, float>& out,
              int n_neighbors,
              const ML::UMAPParams* params,
              cudaStream_t stream)
{
  raft::sparse::selection::brute_force_knn(inputsA.indptr,
                                           inputsA.indices,
                                           inputsA.data,
                                           inputsA.nnz,
                                           inputsA.n,
                                           inputsA.d,
                                           inputsB.indptr,
                                           inputsB.indices,
                                           inputsB.data,
                                           inputsB.nnz,
                                           inputsB.n,
                                           inputsB.d,
                                           out.knn_indices,
                                           out.knn_dists,
                                           n_neighbors,
                                           handle,
                                           ML::Sparse::DEFAULT_BATCH_SIZE,
                                           ML::Sparse::DEFAULT_BATCH_SIZE,
                                           raft::distance::DistanceType::L2Expanded);
}

template <>
void launcher(const raft::handle_t& handle,
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
void launcher(const raft::handle_t& handle,
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
void launcher(const raft::handle_t& handle,
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
