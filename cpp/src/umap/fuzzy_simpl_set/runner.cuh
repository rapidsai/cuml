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

#include <cuml/manifold/umapparams.h>
#include "naive.cuh"

#include <raft/sparse/coo.cuh>

namespace UMAPAlgo {

namespace FuzzySimplSet {

using namespace ML;

/**
 * Calculates a fuzzy simplicial set of the input X and kNN results
 * @param n: number of rows in X
 * @param knn_indices: matrix of kNN indices size (nxn)
 * @param knn_dists: matrix of kNN dists size (nxn)
 * @param n_neighbors number of neighbors
 * @param coo input knn-graph
 * @param params umap parameters
 * @param alloc device allocator
 * @param stream cuda stream
 * @param algorithm algo type to choose
 */
template <int TPB_X, typename value_idx, typename T>
void run(int n, const value_idx *knn_indices, const T *knn_dists,
         int n_neighbors, raft::sparse::COO<T> *coo, UMAPParams *params,
         std::shared_ptr<deviceAllocator> alloc, cudaStream_t stream,
         int algorithm = 0) {
  switch (algorithm) {
    case 0:
      Naive::launcher<TPB_X, value_idx, T>(
        n, knn_indices, knn_dists, n_neighbors, coo, params, alloc, stream);
      break;
  }
}
}  // namespace FuzzySimplSet
};  // namespace UMAPAlgo
