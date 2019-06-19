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

#include "cuML.hpp"
#include "linalg/eltwise.h"
#include "selection/knn.h"
#include "sparse/coo.h"
#define MAX(a, b) ((a < b) ? b : a)

namespace ML {
namespace TSNE {

void get_distances(const float *X, const int n, const int p, long *indices,
                   float *distances, const int n_neighbors,
                   cudaStream_t stream) {
  assert(X != NULL);

  float **knn_input = new float *[1];
  int *sizes = new int[1];
  knn_input[0] = (float *)X;
  sizes[0] = n;

  MLCommon::Selection::brute_force_knn(knn_input, sizes, 1, p,
                                       const_cast<float *>(X), n, indices,
                                       distances, n_neighbors, stream);
  delete knn_input, sizes;
}

float normalize_distances(const int n, float *distances, const size_t SIZE,
                          cudaStream_t stream) {
  // Now D / max(abs(D)) to allow exp(D) to not explode
  assert(distances != NULL);
  thrust::device_ptr<float> begin = thrust::device_pointer_cast(distances);

  float maxNorm = MAX(
    *(thrust::max_element(thrust::cuda::par.on(stream), begin, begin + SIZE)),
    *(thrust::min_element(thrust::cuda::par.on(stream), begin, begin + SIZE)));
  if (maxNorm == 0.0f) maxNorm = 1.0f;

  // Divide distances inplace by max
  const float div = 1.0f / maxNorm;
  MLCommon::LinAlg::scalarMultiply(distances, (const float *)distances, div,
                                   SIZE, stream);
  return div;
}

template <int TPB_X = 32>
void symmetrize_perplexity(float *P, long *indices,
                           MLCommon::Sparse::COO<float> *P_PT, const int n,
                           const int k, const float P_sum,
                           const float exaggeration, cudaStream_t stream,
                           const cumlHandle &handle) {
  assert(P != NULL && indices != NULL);

  // Convert to COO
  MLCommon::Sparse::COO<float> P_COO;
  MLCommon::Sparse::COO<float> P_PT_with_zeros;
  MLCommon::Sparse::from_knn(indices, P, n, k, &P_COO);
  handle.getDeviceAllocator()->deallocate(P, sizeof(float) * n * k, stream);
  handle.getDeviceAllocator()->deallocate(indices, sizeof(long) * n * k,
                                          stream);

  // Perform (P + P.T) / P_sum * early_exaggeration
  const float div = exaggeration / (2.0f * P_sum);
  MLCommon::LinAlg::scalarMultiply(P_COO.vals, (const float *)P_COO.vals, div,
                                   P_COO.nnz, stream);

  // Symmetrize to form P + P.T
  MLCommon::Sparse::coo_symmetrize<TPB_X, float>(
    &P_COO, &P_PT_with_zeros,
    [] __device__(int row, int col, float val, float trans) {
      return val + trans;
    },
    stream);
  P_COO.destroy();

  // Remove all zeros in P + PT
  MLCommon::Sparse::coo_sort<float>(&P_PT_with_zeros, stream);

  MLCommon::Sparse::coo_remove_zeros<TPB_X, float>(&P_PT_with_zeros, P_PT,
                                                   stream);
  P_PT_with_zeros.destroy();
}

}  // namespace TSNE
}  // namespace ML
