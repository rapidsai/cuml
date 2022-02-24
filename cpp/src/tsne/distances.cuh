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

#include <cuml/neighbors/knn_sparse.hpp>
#include <raft/cudart_utils.h>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/eltwise.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.hpp>
#include <raft/sparse/selection/knn.hpp>
#include <selection/knn.cuh>

#include <cuml/manifold/common.hpp>

#include <raft/error.hpp>

#include "utils.cuh"

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace ML {
namespace TSNE {

auto DEFAULT_DISTANCE_METRIC = raft::distance::DistanceType::L2SqrtExpanded;

/**
 * @brief Uses FAISS's KNN to find the top n_neighbors. This speeds up the attractive forces.
 * @param[in] input: dense/sparse manifold input
 * @param[out] indices: The output indices from KNN.
 * @param[out] distances: The output sorted distances from KNN.
 * @param[in] n_neighbors: The number of nearest neighbors you want.
 * @param[in] stream: The GPU stream.
 */
template <typename tsne_input, typename value_idx, typename value_t>
void get_distances(const raft::handle_t& handle,
                   tsne_input& input,
                   knn_graph<value_idx, value_t>& k_graph,
                   cudaStream_t stream);

// dense, int64 indices
template <>
void get_distances(const raft::handle_t& handle,
                   manifold_dense_inputs_t<float>& input,
                   knn_graph<int64_t, float>& k_graph,
                   cudaStream_t stream)
{
  // TODO: for TSNE transform first fit some points then transform with 1/(1+d^2)
  // #861

  std::vector<float*> input_vec = {input.X};
  std::vector<int> sizes_vec    = {input.n};

  /**
 * std::vector<float *> &input, std::vector<int> &sizes,
                     IntType D, float *search_items, IntType n, int64_t *res_I,
                     float *res_D, IntType k,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t userStream,
 */

  raft::spatial::knn::brute_force_knn<int64_t, float, int>(handle,
                                                           input_vec,
                                                           sizes_vec,
                                                           input.d,
                                                           input.X,
                                                           input.n,
                                                           k_graph.knn_indices,
                                                           k_graph.knn_dists,
                                                           k_graph.n_neighbors,
                                                           true,
                                                           true,
                                                           nullptr,
                                                           DEFAULT_DISTANCE_METRIC);
}

// dense, int32 indices
template <>
void get_distances(const raft::handle_t& handle,
                   manifold_dense_inputs_t<float>& input,
                   knn_graph<int, float>& k_graph,
                   cudaStream_t stream)
{
  throw raft::exception("Dense TSNE does not support 32-bit integer indices yet.");
}

// sparse, int32
template <>
void get_distances(const raft::handle_t& handle,
                   manifold_sparse_inputs_t<int, float>& input,
                   knn_graph<int, float>& k_graph,
                   cudaStream_t stream)
{
  raft::sparse::selection::brute_force_knn(input.indptr,
                                           input.indices,
                                           input.data,
                                           input.nnz,
                                           input.n,
                                           input.d,
                                           input.indptr,
                                           input.indices,
                                           input.data,
                                           input.nnz,
                                           input.n,
                                           input.d,
                                           k_graph.knn_indices,
                                           k_graph.knn_dists,
                                           k_graph.n_neighbors,
                                           handle,
                                           ML::Sparse::DEFAULT_BATCH_SIZE,
                                           ML::Sparse::DEFAULT_BATCH_SIZE,
                                           DEFAULT_DISTANCE_METRIC);
}

// sparse, int64
template <>
void get_distances(const raft::handle_t& handle,
                   manifold_sparse_inputs_t<int64_t, float>& input,
                   knn_graph<int64_t, float>& k_graph,
                   cudaStream_t stream)
{
  throw raft::exception("Sparse TSNE does not support 64-bit integer indices yet.");
}

/**
 * @brief   Find the maximum element in the distances matrix, then divide all entries by this.
 *          This promotes exp(distances) to not explode.
 * @param[in] distances: The output sorted distances from KNN
 * @param[in] total_nn: The number of rows in the data X
 * @param[in] stream: The GPU stream
 */
template <typename value_t>
void normalize_distances(value_t* distances, const size_t total_nn, cudaStream_t stream)
{
  auto abs_f      = [] __device__(const value_t& x) { return abs(x); };
  value_t maxNorm = thrust::transform_reduce(rmm::exec_policy(stream),
                                             distances,
                                             distances + total_nn,
                                             abs_f,
                                             0.0f,
                                             thrust::maximum<value_t>());
  raft::linalg::scalarMultiply(distances, distances, 1.0f / maxNorm, total_nn, stream);
}

/**
 * @brief Performs P + P.T.
 * @param[in] P: The perplexity matrix (n, k)
 * @param[in] indices: The input sorted indices from KNN.
 * @param[in] n: The number of rows in the data X.
 * @param[in] k: The number of nearest neighbors.
 * @param[out] COO_Matrix: The final P + P.T output COO matrix.
 * @param[in] stream: The GPU stream.
 * @param[in] handle: The GPU handle.
 */
template <typename value_idx, typename value_t, int TPB_X = 32>
void symmetrize_perplexity(float* P,
                           value_idx* indices,
                           const value_idx n,
                           const int k,
                           const value_t exaggeration,
                           raft::sparse::COO<value_t, value_idx>* COO_Matrix,
                           cudaStream_t stream,
                           const raft::handle_t& handle)
{
  // Symmetrize to form P + P.T
  raft::sparse::linalg::from_knn_symmetrize_matrix<value_idx, value_t>(
    indices, P, n, k, COO_Matrix, stream);
}

}  // namespace TSNE
}  // namespace ML
