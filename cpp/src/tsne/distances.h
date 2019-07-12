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

void get_distances(const float *X, const int n, const int p, long *indices,
                   float *distances, const int n_neighbors,
                   cudaStream_t stream) {
  // TODO: for TSNE transform first fit some points then transform with 1/(1+d^2)
  float **knn_input = new float *[1];
  int *sizes = new int[1];
  knn_input[0] = (float *)X;
  sizes[0] = n;

  MLCommon::Selection::brute_force_knn(knn_input, sizes, 1, p,
                                       const_cast<float *>(X), n, indices,
                                       distances, n_neighbors, stream);
  delete knn_input, sizes;
}

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