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

#include <cuda_utils.h>
#include <iostream>
#include "linalg/unary_op.h"
#include "selection/knn.h"
#include "umap/umapparams.h"

#pragma once

namespace UMAPAlgo {

namespace kNNGraph {

namespace Algo {

using namespace ML;

/**
		 * Initial implementation calls out to FAISS to do its work.
		 * TODO: cuML kNN implementation should support FAISS' approx NN variants (e.g. IVFPQ GPU).
		 */

/**
 * void brute_force_knn(float **input, int *sizes, int n_params, IntType D,
                     float *search_items, IntType n, long *res_I, float *res_D,
                     IntType k, cudaStream_t s)
 */
template <typename T>
void launcher(float *X, int x_n, float *X_query, int x_q_n, int d,
              long *knn_indices, T *knn_dists, int n_neighbors,
              UMAPParams *params, cudaStream_t stream) {
  float **p = new float *[1];
  int *sizes = new int[1];
  p[0] = X;
  sizes[0] = x_n;

  MLCommon::Selection::brute_force_knn(p, sizes, 1, d, X_query, x_q_n,
                                       knn_indices, knn_dists, n_neighbors,
                                       stream);

  MLCommon::LinAlg::unaryOp<T>(
    knn_dists, knn_dists, x_n * n_neighbors,
    [] __device__(T input) { return sqrt(input); }, stream);

  delete p;
  delete sizes;
}
}  // namespace Algo
}  // namespace kNNGraph
};  // namespace UMAPAlgo
