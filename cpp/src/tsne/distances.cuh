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

#include "utils.cuh"

#include <cuml/common/distance_type.hpp>
#include <cuml/manifold/common.hpp>
#include <cuml/neighbors/knn_sparse.hpp>

#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <selection/knn.cuh>

namespace ML {
namespace TSNE {

/**
 * @brief Uses CUVS's KNN to find the top n_neighbors. This speeds up the attractive forces.
 * @param[in] input: dense/sparse manifold input
 * @param[out] indices: The output indices from KNN.
 * @param[out] distances: The output sorted distances from KNN.
 * @param[in] n_neighbors: The number of nearest neighbors you want.
 * @param[in] stream: The GPU stream.
 * @param[in] metric: The distance metric.
 */
template <typename tsne_input, typename value_idx, typename value_t>
void get_distances(const raft::handle_t& handle,
                   tsne_input& input,
                   knn_graph<value_idx, value_t>& k_graph,
                   cudaStream_t stream,
                   ML::distance::DistanceType metric,
                   value_t p);

// dense, int64 indices
template <>
void get_distances(const raft::handle_t& handle,
                   manifold_dense_inputs_t<float>& input,
                   knn_graph<int64_t, float>& k_graph,
                   cudaStream_t stream,
                   ML::distance::DistanceType metric,
                   float p)
{
  // TODO: for TSNE transform first fit some points then transform with 1/(1+d^2)
  // #861
  auto k = k_graph.n_neighbors;
  auto X_view =
    raft::make_device_matrix_view<const float, int64_t, raft::col_major>(input.X, input.n, input.d);
  auto idx = cuvs::neighbors::brute_force::build(
    handle, X_view, static_cast<cuvs::distance::DistanceType>(metric), p);

  cuvs::neighbors::brute_force::search(
    handle,
    idx,
    X_view,
    raft::make_device_matrix_view<int64_t, int64_t>(k_graph.knn_indices, input.n, k),
    raft::make_device_matrix_view<float, int64_t>(k_graph.knn_dists, input.n, k));
}

// dense, int32 indices
template <>
void get_distances(const raft::handle_t& handle,
                   manifold_dense_inputs_t<float>& input,
                   knn_graph<int, float>& k_graph,
                   cudaStream_t stream,
                   ML::distance::DistanceType metric,
                   float p)
{
  throw raft::exception("Dense TSNE does not support 32-bit integer indices yet.");
}

// sparse, int32
template <>
void get_distances(const raft::handle_t& handle,
                   manifold_sparse_inputs_t<int, float>& input,
                   knn_graph<int, float>& k_graph,
                   cudaStream_t stream,
                   ML::distance::DistanceType metric,
                   float p)
{
  auto input_structure = raft::make_device_compressed_structure_view<int, int, int>(
    input.indptr, input.indices, input.n, input.d, input.nnz);
  auto input_csr = raft::make_device_csr_matrix_view<const float>(input.data, input_structure);

  cuvs::neighbors::brute_force::sparse_search_params search_params;
  search_params.batch_size_index = ML::Sparse::DEFAULT_BATCH_SIZE;
  search_params.batch_size_query = ML::Sparse::DEFAULT_BATCH_SIZE;

  auto index = cuvs::neighbors::brute_force::build(
    handle, input_csr, static_cast<cuvs::distance::DistanceType>(metric), p);

  cuvs::neighbors::brute_force::search(
    handle,
    search_params,
    index,
    input_csr,
    raft::make_device_matrix_view<int, int64_t>(k_graph.knn_indices, input.n, k_graph.n_neighbors),
    raft::make_device_matrix_view<float, int64_t>(k_graph.knn_dists, input.n, k_graph.n_neighbors));
}

// sparse, int64
template <>
void get_distances(const raft::handle_t& handle,
                   manifold_sparse_inputs_t<int64_t, float>& input,
                   knn_graph<int64_t, float>& k_graph,
                   cudaStream_t stream,
                   ML::distance::DistanceType metric,
                   float p)
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
  auto abs_f = cuda::proclaim_return_type<value_t>(
    [] __device__(const value_t& x) -> value_t { return abs(x); });
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
