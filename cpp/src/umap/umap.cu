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

#include "umap.cuh"

#include <rmm/device_buffer.hpp>

namespace ML {
namespace UMAP {

void find_ab(const raft::handle_t& handle, UMAPParams* params)
{
  cudaStream_t stream = handle.get_stream();
  UMAPAlgo::find_ab(params, stream);
}

std::unique_ptr<raft::sparse::COO<float, int>> get_graph(
  const raft::handle_t& handle,
  float* X,  // input matrix
  float* y,  // labels
  int n,
  int d,
  knn_indices_dense_t* knn_indices,  // precomputed indices
  float* knn_dists,                  // precomputed distances
  UMAPParams* params)
{
  if (dispatch_to_uint64_t(n, params->n_neighbors, params->n_components))
    return _get_graph<uint64_t>(handle, X, y, n, d, knn_indices, knn_dists, params);
  else
    return _get_graph<int>(handle, X, y, n, d, knn_indices, knn_dists, params);
}

void refine(const raft::handle_t& handle,
            float* X,
            int n,
            int d,
            raft::sparse::COO<float>* graph,
            UMAPParams* params,
            float* embeddings)
{
  if (dispatch_to_uint64_t(n, params->n_neighbors, params->n_components))
    _refine<uint64_t>(handle, X, n, d, graph, params, embeddings);
  else
    _refine<int>(handle, X, n, d, graph, params, embeddings);
}

void init_and_refine(const raft::handle_t& handle,
                     float* X,
                     int n,
                     int d,
                     raft::sparse::COO<float>* graph,
                     UMAPParams* params,
                     float* embeddings)
{
  if (dispatch_to_uint64_t(n, params->n_neighbors, params->n_components))
    _init_and_refine<uint64_t>(handle, X, n, d, graph, params, embeddings);
  else
    _init_and_refine<int>(handle, X, n, d, graph, params, embeddings);
}

void fit(const raft::handle_t& handle,
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
  if (dispatch_to_uint64_t(n, params->n_neighbors, params->n_components))
    _fit<uint64_t>(handle, X, y, n, d, knn_indices, knn_dists, params, embeddings, graph);
  else
    _fit<int>(handle, X, y, n, d, knn_indices, knn_dists, params, embeddings, graph);
}

void fit_sparse(const raft::handle_t& handle,
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
  if (dispatch_to_uint64_t(n, params->n_neighbors, params->n_components))
    _fit_sparse<uint64_t>(handle,
                          indptr,
                          indices,
                          data,
                          nnz,
                          y,
                          n,
                          d,
                          knn_indices,
                          knn_dists,
                          params,
                          embeddings,
                          graph);
  else
    _fit_sparse<int>(handle,
                     indptr,
                     indices,
                     data,
                     nnz,
                     y,
                     n,
                     d,
                     knn_indices,
                     knn_dists,
                     params,
                     embeddings,
                     graph);
}

void transform(const raft::handle_t& handle,
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
  if (dispatch_to_uint64_t(n, params->n_neighbors, params->n_components))
    _transform<uint64_t>(
      handle, X, n, d, orig_X, orig_n, embedding, embedding_n, params, transformed);
  else
    _transform<int>(handle, X, n, d, orig_X, orig_n, embedding, embedding_n, params, transformed);
}

void transform_sparse(const raft::handle_t& handle,
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
  if (dispatch_to_uint64_t(n, params->n_neighbors, params->n_components))
    _transform_sparse<uint64_t>(handle,
                                indptr,
                                indices,
                                data,
                                nnz,
                                n,
                                d,
                                orig_x_indptr,
                                orig_x_indices,
                                orig_x_data,
                                orig_nnz,
                                orig_n,
                                embedding,
                                embedding_n,
                                params,
                                transformed);
  else
    _transform_sparse<int>(handle,
                           indptr,
                           indices,
                           data,
                           nnz,
                           n,
                           d,
                           orig_x_indptr,
                           orig_x_indices,
                           orig_x_data,
                           orig_nnz,
                           orig_n,
                           embedding,
                           embedding_n,
                           params,
                           transformed);
}

}  // namespace UMAP
}  // namespace ML
