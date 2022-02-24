/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/sparse/coo.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace raft {
class handle_t;
}  // namespace raft

namespace ML {
class UMAPParams;
namespace UMAP {

void transform(const raft::handle_t& handle,
               float* X,
               int n,
               int d,
               int64_t* knn_indices,
               float* knn_dists,
               float* orig_X,
               int orig_n,
               float* embedding,
               int embedding_n,
               UMAPParams* params,
               float* transformed);

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

void find_ab(const raft::handle_t& handle, UMAPParams* params);

void fit(const raft::handle_t& handle,
         float* X,  // input matrix
         float* y,  // labels
         int n,
         int d,
         int64_t* knn_indices,
         float* knn_dists,
         UMAPParams* params,
         float* embeddings);

/**
 * refine performs a UMAP fit on existing embeddings without reinitializing them, which enables
 * iterative fitting without callbacks.
 *
 * @param handle: raft::handle_t
 * @param X: pointer to input array
 * @param n: n_samples of input array
 * @param d: n_features of input array
 * @param cgraph_coo: pointer to raft::sparse::COO object computed using ML::UMAP::get_graph
 * @param params: pointer to ML::UMAPParams object
 * @param embeddings: pointer to current embedding with shape n * n_components, stores updated
 * embeddings on executing refine
 */
void refine(const raft::handle_t& handle,
            float* X,  // input matrix
            int n,
            int d,
            raft::sparse::COO<float, int>* cgraph_coo,
            UMAPParams* params,
            float* embeddings);

/**
 * returns a simplical set as a raft::sparse:COO object to be consumed by the ML::UMAP::refine
 * function.
 *
 * @param handle: raft::handle_t
 * @param X: pointer to input array
 * @param y: pointer to labels array
 * @param n: n_samples of input array
 * @param d: n_features of input array
 * @param params: pointer to ML::UMAPParams object
 * @return: simplical set (pointer to raft::sparse::COO object)
 */
std::unique_ptr<raft::sparse::COO<float, int>> get_graph(const raft::handle_t& handle,
                                                         float* X,  // input matrix
                                                         float* y,  // labels
                                                         int n,
                                                         int d,
                                                         UMAPParams* params);

void fit_sparse(const raft::handle_t& handle,
                int* indptr,  // input matrix
                int* indices,
                float* data,
                size_t nnz,
                float* y,
                int n,  // rows
                int d,  // cols
                UMAPParams* params,
                float* embeddings);
}  // namespace UMAP
}  // namespace ML
