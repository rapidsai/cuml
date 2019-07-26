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

#pragma once

#include <linalg/eltwise.h>
#include <selection/knn.h>
#include "sparse/coo.h"
#include "utils.h"

namespace ML {
namespace TSNE {

/**
 * @brief Uses FAISS's KNN to find the top n_neighbors. This speeds up the attractive forces.
 * @input param X: The GPU handle.
 * @input param n: The number of rows in the data X.
 * @input param p: The number of columns in the data X.
 * @output param indices: The output indices from KNN.
 * @output param distances: The output sorted distances from KNN.
 * @input param n_neighbors: The number of nearest neighbors you want.
 * @input param stream: The GPU stream.
 */
void get_distances(const float *X, const int n, const int p, long *indices,
                   float *distances, const int n_neighbors,
                   cudaStream_t stream) {
  // TODO: for TSNE transform first fit some points then transform with 1/(1+d^2)
  // #861
  float **knn_input = new float *[1];
  int *sizes = new int[1];
  knn_input[0] = (float *)X;
  sizes[0] = n;

  MLCommon::Selection::brute_force_knn(knn_input, sizes, 1, p,
                                       const_cast<float *>(X), n, indices,
                                       distances, n_neighbors, stream);
  delete knn_input, sizes;
}

/**
 * @brief   Find the maximum element in the distances matrix, then divide all entries by this.
 *          This promotes exp(distances) to not explode.
 * @input param n: The number of rows in the data X.
 * @input param distances: The output sorted distances from KNN.
 * @input param n_neighbors: The number of nearest neighbors you want.
 * @input param stream: The GPU stream.
 */
void normalize_distances(const int n, float *distances, const int n_neighbors,
                         cudaStream_t stream) {
  // Now D / max(abs(D)) to allow exp(D) to not explode
  thrust::device_ptr<float> begin = thrust::device_pointer_cast(distances);
  float maxNorm = *thrust::max_element(thrust::cuda::par.on(stream), begin,
                                       begin + n * n_neighbors);
  if (maxNorm == 0.0f) maxNorm = 1.0f;

  // Divide distances inplace by max
  const float div = 1.0f / maxNorm;  // Mult faster than div
  MLCommon::LinAlg::scalarMultiply(distances, distances, div, n * n_neighbors,
                                   stream);
}

/**
 * @brief Performs P + P.T.
 * @input param P: The perplexity matrix (n, k)
 * @input param indices: The input sorted indices from KNN.
 * @input param n: The number of rows in the data X.
 * @input param k: The number of nearest neighbors you want.
 * @input param P_sum: The sum of P.
 * @input param exaggeration: How much early pressure you want the clusters in TSNE to spread out more.
 * @output param COO_Matrix: The final P + P.T output COO matrix.
 * @input param stream: The GPU stream.
 * @input param handle: The GPU handle.
 */
template <int TPB_X = 32>
void symmetrize_perplexity(float *P, long *indices, const int n, const int k,
                           const float P_sum, const float exaggeration,
                           MLCommon::Sparse::COO<float> *COO_Matrix,
                           cudaStream_t stream, const cumlHandle &handle) {
  // Perform (P + P.T) / P_sum * early_exaggeration
  const float div = exaggeration / (2.0f * P_sum);
  MLCommon::LinAlg::scalarMultiply(P, P, div, n * k, stream);

  // Symmetrize to form P + P.T
  // struct COO_Matrix_t COO_Matrix = symmetrize_matrix(P, indices, n, k, handle);
  MLCommon::Sparse::from_knn_symmetrize_matrix(indices, P, n, k, COO_Matrix,
                                               stream);

  handle.getDeviceAllocator()->deallocate(P, sizeof(float) * n * k, stream);
  handle.getDeviceAllocator()->deallocate(indices, sizeof(long) * n * k,
                                          stream);
}

}  // namespace TSNE
}  // namespace ML