/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "algo.h"

#pragma once

namespace UMAPAlgo {

namespace kNNGraph {

using namespace ML;

/**
  * @brief This function performs a k-nearest neighbors against
  *        the input algorithm using the specified knn algorithm. 
  *        Only algorithm supported at the moment is brute force
  *        knn primitive. 
  * @param X: Matrix to query (size n x d) in row-major format
  * @param n: Number of rows in X
  * @param query: Search matrix in row-major format
  * @param q_n: Number of rows in query matrix
  * @param d: Number of columns in X and query matrices
  * @param knn_indices: Return indices matrix (size n*k)
  * @param knn_dists: Return dists matrix (size n*k)
  * @param n_neighbors: Number of closest neighbors, k, to query
  * @param params: Instance of UMAPParam settings
  * @param stream: cuda stream to use
  * @param algo: Algorithm to use. Currently only brute force is supported
  * @tpatam T: Type of input, query, and dist matrices. Usually float
 */
template <typename T = float>
void run(T *X, int n, T *query, int q_n, int d, long *knn_indices, T *knn_dists,
         int n_neighbors, UMAPParams *params, cudaStream_t stream,
         int algo = 0) {
  switch (algo) {
    /**
      * Initial algo uses FAISS indices
      */
    case 0:
      Algo::launcher(X, n, query, q_n, d, knn_indices, knn_dists, n_neighbors,
                     params, stream);
      break;
  }
}
}  // namespace kNNGraph
};  // namespace UMAPAlgo
