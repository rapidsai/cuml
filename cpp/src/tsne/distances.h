
#pragma once

#include <linalg/eltwise.h>
#include <selection/knn.h>
#include "cuML.hpp"
#include "kernels.h"
#include "utils.h"

namespace ML {
using namespace MLCommon;

void get_distances(const float *X, const int n, const int p, long *indices,
                   float *distances, const int n_neighbors,
                   cuda_stream_t stream) {
  cumlHandle handle;

  float **knn_input = new float *[1];
  int *sizes = new int[1];
  knn_input[0] = (float *)X;
  sizes[0] = n;

  MLCommon::Selection::brute_force_knn(knn_input, sizes, 1, p,
                                       const_cast<float *>(X), n, indices,
                                       distances, n_neighbors, stream);

  // Now D / max(abs(D)) to allow exp(D) to not explode
  thrust_t<float> begin = to_thrust(distances);
  thrust_t<float> end = begin + n * n_neighbors;

  float maxNorm = MAX(*(thrust::max_element(__STREAM__, begin, end)),
                      *(thrust::min_element(__STREAM__, begin, end)));
  if (maxNorm == 0.0f) maxNorm = 1.0f;

  // Divide distances inplace by max
  float div_maxNorm = 1.0f / maxNorm;  // Mult faster than div
  LinAlg::scalarMultiply(distances, distances, div_maxNorm, n * n_neighbors,
                         stream);

  // Remove temp variables
  delete knn_input, sizes;
}

void symmetrize_perplexity(float *P, long *indices, COO_t<float> *P_PT,
                           const int n, const int k, const float P_sum,
                           const float exaggeration, cuda_stream_t stream) {
  // Convert to COO
  COO_t<float> P_COO;
  Sparse::from_knn(indices, P, n, k, &P_COO);
  cfree(P);
  cfree(indices);

  // Perform (P + P.T) / P_sum * early_exaggeration
  Sparse::coo_symmetrize<32, float>(
    &P_COO, P_PT,
    [] __device__(int row, int col, float val, float trans) {
      return val + trans;
    },
    stream);
  cudaDeviceSynchronize();

  // Divide by P_sum
  // Notice P_sum is *2 since symmetric.
  const float div = exaggeration / (2.0f * P_sum);

  inplace_multiply(P_PT->vals, P_PT->nnz, div);

  P_COO.destroy();
}

}  // namespace ML
