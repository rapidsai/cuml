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

#pragma once

#include <common/cudart_utils.h>
#include <linalg/eltwise.cuh>
#include <selection/knn.cuh>
#include <sparse/coo.cuh>

namespace ML {
namespace TSNE {

/**
 * @brief Uses FAISS's KNN to find the top n_neighbors. This speeds up the attractive forces.
 * @param[in] X: The GPU handle.
 * @param[in] n: The number of rows in the data X.
 * @param[in] p: The number of columns in the data X.
 * @param[out] indices: The output indices from KNN.
 * @param[out] distances: The output sorted distances from KNN.
 * @param[in] n_neighbors: The number of nearest neighbors you want.
 * @param[in] d_alloc: device allocator
 * @param[in] stream: The GPU stream.
 */
void get_distances(const float *X, const int n, const int p, long *indices,
                   float *distances, const int n_neighbors,
                   std::shared_ptr<deviceAllocator> d_alloc,
                   cudaStream_t stream) {
  // TODO: for TSNE transform first fit some points then transform with 1/(1+d^2)
  // #861

  std::vector<float *> input_vec = {const_cast<float *>(X)};
  std::vector<int> sizes_vec = {n};

  /**
 * std::vector<float *> &input, std::vector<int> &sizes,
                     IntType D, float *search_items, IntType n, int64_t *res_I,
                     float *res_D, IntType k,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t userStream,
 */

  MLCommon::Selection::brute_force_knn(input_vec, sizes_vec, p,
                                       const_cast<float *>(X), n, indices,
                                       distances, n_neighbors, d_alloc, stream);
}

/**
 * @brief   Find the maximum element in the distances matrix, then divide all entries by this.
 *          This promotes exp(distances) to not explode.
 * @param[in] n: The number of rows in the data X.
 * @param[in] distances: The output sorted distances from KNN.
 * @param[in] n_neighbors: The number of nearest neighbors you want.
 * @param[in] stream: The GPU stream.
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
  raft::linalg::scalarMultiply(distances, distances, div, n * n_neighbors,
                               stream);
}

/**
 * @brief Performs P + P.T.
 * @param[in] P: The perplexity matrix (n, k)
 * @param[in] indices: The input sorted indices from KNN.
 * @param[in] n: The number of rows in the data X.
 * @param[in] k: The number of nearest neighbors.
 * @param[in] exaggeration: How much early pressure you want the clusters in TSNE to spread out more.
 * @param[out] COO_Matrix: The final P + P.T output COO matrix.
 * @param[in] stream: The GPU stream.
 * @param[in] handle: The GPU handle.
 */
void symmetrize_perplexity(float *P, long *indices, const int n, const int k,
                           const float exaggeration,
                           MLCommon::Sparse::COO<float> *COO_Matrix,
                           cudaStream_t stream, const raft::handle_t &handle) {
  // Perform (P + P.T) / P_sum * early_exaggeration
  const float div = exaggeration / (2.0f * n);
  raft::linalg::scalarMultiply(P, P, div, n * k, stream);

  // Symmetrize to form P + P.T
  MLCommon::Sparse::from_knn_symmetrize_matrix(
    indices, P, n, k, COO_Matrix, stream, handle.get_device_allocator());
}

}  // namespace TSNE
}  // namespace ML
