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

#include <cuml/manifold/umapparams.h>

#include <raft/core/host_coo_matrix.hpp>
#include <raft/sparse/coo.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace raft {
class handle_t;
}  // namespace raft

namespace ML {
class UMAPParams;
namespace UMAP {

/**
 * Returns the simplical set to be consumed by the ML::UMAP::refine function.
 *
 * @param[in] handle: raft::handle_t
 * @param[out] params: pointer to ML::UMAPParams object of which the a and b parameters will be
 * updated
 */
void find_ab(const raft::handle_t& handle, UMAPParams* params);

/**
 * Returns the simplical set to be consumed by the ML::UMAP::refine function.
 *
 * @param[in] handle: raft::handle_t
 * @param[in] X: pointer to input array
 * @param[in] y: pointer to labels array
 * @param[in] n: n_samples of input array
 * @param[in] d: n_features of input array
 * @param[in] knn_indices: pointer to knn_indices (optional)
 * @param[in] knn_dists: pointer to knn_dists (optional)
 * @param[in] params: pointer to ML::UMAPParams object
 * @return: simplical set as a unique pointer to a raft::sparse::COO object
 */
std::unique_ptr<raft::sparse::COO<float, int>> get_graph(const raft::handle_t& handle,
                                                         float* X,  // input matrix
                                                         float* y,  // labels
                                                         int n,
                                                         int d,
                                                         int64_t* knn_indices,
                                                         float* knn_dists,
                                                         UMAPParams* params);

/**
 * Performs a UMAP fit on existing embeddings without reinitializing them, which enables
 * iterative fitting without callbacks.
 *
 * @param[in] handle: raft::handle_t
 * @param[in] X: pointer to input array
 * @param[in] n: n_samples of input array
 * @param[in] d: n_features of input array
 * @param[in] graph: pointer to raft::sparse::COO object computed using ML::UMAP::get_graph
 * @param[in] params: pointer to ML::UMAPParams object
 * @param[out] embeddings: pointer to current embedding with shape n * n_components, stores updated
 * embeddings on executing refine
 */
void refine(const raft::handle_t& handle,
            float* X,
            int n,
            int d,
            raft::sparse::COO<float, int>* graph,
            UMAPParams* params,
            float* embeddings);

/**
 * Initializes embeddings and performs a UMAP fit on them, which enables
 * iterative fitting without callbacks.
 *
 * @param[in] handle: raft::handle_t
 * @param[in] X: pointer to input array
 * @param[in] n: n_samples of input array
 * @param[in] d: n_features of input array
 * @param[in] graph: pointer to raft::sparse::COO object computed using ML::UMAP::get_graph
 * @param[in] params: pointer to ML::UMAPParams object
 * @param[out] embeddings: pointer to current embedding with shape n * n_components, stores updated
 * embeddings on executing refine
 */
void init_and_refine(const raft::handle_t& handle,
                     float* X,
                     int n,
                     int d,
                     raft::sparse::COO<float, int>* graph,
                     UMAPParams* params,
                     float* embeddings);

/**
 * Dense fit
 *
 * @param[in] handle: raft::handle_t
 * @param[in] X: pointer to input array
 * @param[in] y: pointer to labels array
 * @param[in] n: n_samples of input array
 * @param[in] d: n_features of input array
 * @param[in] knn_indices: pointer to knn_indices of input (optional)
 * @param[in] knn_dists: pointer to knn_dists of input (optional)
 * @param[in] params: pointer to ML::UMAPParams object
 * @param[out] embeddings: unique_ptr to device_buffer that will be allocated and filled with
 * embeddings
 * @param[out] graph: pointer to fuzzy simplicial set graph
 */
void fit(const raft::handle_t& handle,
         float* X,
         float* y,
         int n,
         int d,
         int64_t* knn_indices,
         float* knn_dists,
         UMAPParams* params,
         std::unique_ptr<rmm::device_buffer>& embeddings,
         raft::host_coo_matrix<float, int, int, uint64_t>& graph);

/**
 * Sparse fit
 *
 * @param[in] handle: raft::handle_t
 * @param[in] indptr: pointer to index pointer array of input array
 * @param[in] indices: pointer to index array of input array
 * @param[in] data: pointer to data array of input array
 * @param[in] nnz: pointer to data array of input array
 * @param[in] y: pointer to labels array
 * @param[in] n: n_samples of input array
 * @param[in] d: n_features of input array
 * @param[in] knn_indices: pointer to knn_indices of input (optional)
 * @param[in] knn_dists: pointer to knn_dists of input (optional)
 * @param[in] params: pointer to ML::UMAPParams object
 * @param[out] embeddings: unique_ptr to device_buffer that will be allocated and filled with
 * embeddings
 * @param[out] graph: pointer to fuzzy simplicial set graph
 */
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
                raft::host_coo_matrix<float, int, int, uint64_t>& graph);

/**
 * Dense transform
 *
 * @param[in] handle: raft::handle_t
 * @param[in] X: pointer to input array to be inferred
 * @param[in] n: n_samples of input array to be inferred
 * @param[in] d: n_features of input array to be inferred
 * @param[in] orig_X: pointer to original training array
 * @param[in] orig_n: number of rows in original training array
 * @param[in] embedding: pointer to embedding created during training
 * @param[in] embedding_n: number of rows in embedding created during training
 * @param[in] params: pointer to ML::UMAPParams object
 * @param[out] transformed: pointer to embedding produced through projection
 */
void transform(const raft::handle_t& handle,
               float* X,
               int n,
               int d,
               float* orig_X,
               int orig_n,
               float* embedding,
               int embedding_n,
               UMAPParams* params,
               float* transformed);

/**
 * Sparse transform
 *
 * @param[in] handle: raft::handle_t
 * @param[in] indptr: pointer to index pointer array of input array to be inferred
 * @param[in] indices: pointer to index array of input array to be inferred
 * @param[in] data: pointer to data array of input array to be inferred
 * @param[in] nnz: number of stored values of input array to be inferred
 * @param[in] n: n_samples of input array
 * @param[in] d: n_features of input array
 * @param[in] orig_x_indptr: pointer to index pointer array of original training array
 * @param[in] orig_x_indices: pointer to index array of original training array
 * @param[in] orig_x_data: pointer to data array of original training array
 * @param[in] orig_nnz: number of stored values of original training array
 * @param[in] orig_n: number of rows in original training array
 * @param[in] embedding: pointer to embedding created during training
 * @param[in] embedding_n: number of rows in embedding created during training
 * @param[in] params: pointer to ML::UMAPParams object
 * @param[out] transformed: pointer to embedding produced through projection
 */
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
                      float* transformed);

}  // namespace UMAP
}  // namespace ML
