/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuml/manifold/umapparams.h>
#include <iostream>
#include <linalg/unary_op.cuh>
#include <selection/knn.cuh>
#include <sparse/knn.cuh>

#include <raft/sparse/cusparse_wrappers.h>

#pragma once

namespace UMAPAlgo {

namespace kNNGraph {

namespace Algo {

using namespace ML;

/**
 * Initial implementation calls out to FAISS to do its work.
 */

/**
 * void brute_force_knn(float **input, int *sizes, int n_params, IntType D,
                     float *search_items, IntType n, int64_t *res_I, float *res_D,
                     IntType k, cudaStream_t s)
 */


template<typename T, typename umap_inputs>
void launcher(umap_inputs &inputsA, umap_inputs &inputsB,
              int64_t **knn_indices, T **knn_dists, int n_neighbors,
              UMAPParams *params, std::shared_ptr<deviceAllocator> d_alloc,
              cudaStream_t stream);

void launcher(umap_dense_inputs_t<float> &inputsA, umap_dense_inputs_t<float> &inputsB,
              int64_t **knn_indices, float **knn_dists, int n_neighbors,
              UMAPParams *params, std::shared_ptr<deviceAllocator> d_alloc,
              cudaStream_t stream) {
  std::vector<float *> ptrs(1);
  std::vector<int> sizes(1);
  ptrs[0] = inputsA.X;
  sizes[0] = inputsA.n;

  MLCommon::Selection::brute_force_knn(ptrs, sizes, inputsA.d, inputsB.X, inputsB.n,
                                       *knn_indices, *knn_dists, n_neighbors,
                                       d_alloc, stream);
}

template<>
void launcher(umap_sparse_inputs_t<int, float> &inputsA,
              umap_sparse_inputs_t<int, float> &inputsB,
              int64_t **knn_indices, float **knn_dists, int n_neighbors,
              UMAPParams *params, std::shared_ptr<deviceAllocator> d_alloc,
              cudaStream_t stream) {

  // TODO: Use handle as input here and remove manual creation
  cusparseHandle_t cusparseHandle;
  CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

//  MLCommon::Sparse::Selection::brute_force_knn(
//    inputsA.indptr, inputsA.indices, inputsA.data, inputsA.nnz, inputsA.n,
//    inputsA.d, inputsB.indptr, inputsB.indices, inputsB.data, inputsB.nnz,
//    inputsB.n, inputsB.d, *knn_indices, *knn_dists, n_neighbors, cusparseHandle,
//    d_alloc, stream, ML::MetricType::METRIC_L2);

  CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));
}

template<>
inline void launcher(umap_precomputed_knn_inputs_t<float> &inputsA,
              umap_precomputed_knn_inputs_t<float> &inputsB,
              int64_t **knn_indices, float **knn_dists, int n_neighbors,
              UMAPParams *params, std::shared_ptr<deviceAllocator> d_alloc,
              cudaStream_t stream) {

  *knn_indices = inputsA.knn_indices;
  *knn_dists = inputsA.knn_dists;
}

}  // namespace Algo
}  // namespace kNNGraph
};  // namespace UMAPAlgo
