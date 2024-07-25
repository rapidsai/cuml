/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "algo.cuh"

#include <cuml/manifold/common.hpp>

namespace UMAPAlgo {

namespace kNNGraph {

using namespace ML;

/**
 * @brief This function performs a k-nearest neighbors against
 *        the input algorithm using the specified knn algorithm.
 *        Only algorithm supported at the moment is brute force
 *        knn primitive.
 * @tparam value_idx: Type of knn indices matrix. Usually an integral type.
 * @tparam value_t: Type of input, query, and dist matrices. Usually float
 * @param[in] X: Matrix to query (size n x d) in row-major format
 * @param[in] n: Number of rows in X
 * @param[in] query: Search matrix in row-major format
 * @param[in] q_n: Number of rows in query matrix
 * @param[in] d: Number of columns in X and query matrices
 * @param[out] knn_graph : output knn_indices and knn_dists (size n*k)
 * @param[in] n_neighbors: Number of closest neighbors, k, to query
 * @param[in] params: Instance of UMAPParam settings
 * @param[in] stream: cuda stream to use
 * @param[in] algo: Algorithm to use. Currently only brute force is supported
 */
template <typename value_idx = int64_t, typename value_t = float, typename umap_inputs>
void run(const raft::handle_t& handle,
         const umap_inputs& inputsA,
         const umap_inputs& inputsB,
         knn_graph<value_idx, value_t>& out,
         int n_neighbors,
         const UMAPParams* params,
         cudaStream_t stream,
         int algo = 0)
{
  switch (algo) {
    /**
     * Initial algo uses FAISS indices
     */
    case 0:
      Algo::launcher<value_idx, value_t, umap_inputs>(
        handle, inputsA, inputsB, out, n_neighbors, params, stream);
      break;
  }
}

}  // namespace kNNGraph
};  // namespace UMAPAlgo
