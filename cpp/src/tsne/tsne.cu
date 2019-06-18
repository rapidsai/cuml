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

#include "common/cumlHandle.hpp"

#include "distances.h"
#include "fast_kernels.h"
#include "linalg/norm.h"
#include "tsne/tsne.h"
#include "utils.h"

namespace ML {
using MLCommon::ceildiv;

void TSNE_fit(const cumlHandle &handle, const float *X, float *Y, const int n,
              const int p, const int n_components, int n_neighbors,

              float perplexity, const int perplexity_max_iter,
              const int perplexity_tol,

              const float early_exaggeration, const int exaggeration_iter,

              const float min_gain, const float gains_add,
              const float gains_mult, const float eta, const int max_iter,
              const float pre_momentum, const float post_momentum,

              const long long seed, const bool initialize_embeddings,
              const bool verbose)
// Method = 0 for Naive, 1 for Fast
{
  assert(n > 0 && p > 0 && n_components > 0 && n_neighbors > 0 && X != NULL &&
         Y != NULL);
  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  if (n_neighbors > n) {
    printf(
      "[Warn]  Notice n = %d, n_neighbors = %d. n_neighbors must be <= n!\n", n,
      n_neighbors);
    n_neighbors = n;
  }
  if (perplexity >= n) {
    printf("[Warn]  Notice n = %d, perplexity = %f. perplexity must be <= n!\n",
           n, perplexity);
    perplexity = n;
  }
  if (verbose)
    printf(
      "[Info]  Data = (%d, %d) with n_components = %d and perplexity = %f\n", n,
      p, n_components, perplexity);

  // Get distances
  if (verbose) printf("[Info] Getting distances.\n");

  float *distances =
    (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);
  long *indices =
    (long *)d_alloc->allocate(sizeof(long) * n * n_neighbors, stream);

  TSNE::get_distances(X, n, p, indices, distances, n_neighbors, stream);

  // Normalize distances
  if (verbose)
    printf("[Info] Now normalizing distances so exp(D) doesn't explode.\n");
  TSNE::normalize_distances(n, distances, n * n_neighbors, stream);

  // Optimal perplexity
  if (verbose)
    printf("[Info] Searching for optimal perplexity via bisection search.\n");
  float *P =
    (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);

  // Determine best blocksize / gridsize
  int blockSize_N = 1024;  // default to 1024
  int minGridSize_N;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize_N, &blockSize_N,
                                     TSNE::__determine_sigmas, 0, n);
  const int gridSize_N = ceildiv(n, blockSize_N);

  const float P_sum = TSNE::determine_sigmas(
    distances, P, perplexity, perplexity_max_iter, perplexity_tol, n,
    n_neighbors, stream, gridSize_N, blockSize_N, handle);
  d_alloc->deallocate(distances, sizeof(float) * n * n_neighbors, stream);
  if (verbose) printf("[Info] Perplexity sum = %f\n", P_sum);

  // Convert data to COO layout
  MLCommon::Sparse::COO<float> P_PT;
  TSNE::symmetrize_perplexity(P, indices, &P_PT, n, n_neighbors, P_sum,
                              early_exaggeration, stream, handle);

  const int NNZ = P_PT.nnz;
  float *VAL = P_PT.vals;
  const int *COL = P_PT.rows;
  const int *ROW = P_PT.cols;

  // Allocate data [NOTICE Fortran Contiguous for method = Naive and C-Contiguous for fast]
  if (initialize_embeddings)
    TSNE::random_vector(Y, -0.03f, 0.03f, n * n_components, stream, seed);

  // Allocate space
  if (verbose) printf("[Info] Now allocating memory for TSNE.\n");
  float *norm = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  float *Q_sum = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  double *sum = (double *)d_alloc->allocate(sizeof(double), stream);

  float *attract =
    (float *)d_alloc->allocate(sizeof(float) * n * n_components, stream);
  float *repel =
    (float *)d_alloc->allocate(sizeof(float) * n * n_components, stream);

  float *iY =
    (float *)d_alloc->allocate(sizeof(float) * n * n_components, stream);
  float *gains =
    (float *)d_alloc->allocate(sizeof(float) * n * n_components, stream);
  float *means =
    (float *)d_alloc->allocate(sizeof(float) * n_components, stream);

  // Compute optimal gridSize and blockSize for attractive forces
  int blockSize_NNZ = 1024;  // default to 1024
  int minGridSize_NNZ;
  if (n_components == 2)
    cudaOccupancyMaxPotentialBlockSize(&minGridSize_NNZ, &blockSize_NNZ,
                                       TSNE::__attractive_fast_2dim, 0, NNZ);
  else
    cudaOccupancyMaxPotentialBlockSize(&minGridSize_NNZ, &blockSize_NNZ,
                                       TSNE::__attractive_fast, 0, NNZ);
  const int gridSize_NNZ = ceildiv(NNZ, blockSize_NNZ);

  // Compute optimal gridSize and blockSize for applying forces
  int blockSize_dimN = 1024;  // default to 1024
  int minGridSize_dimN;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize_dimN, &blockSize_dimN,
                                     TSNE::__apply_forces, 0, n * n_components);
  const int gridSize_dimN = ceildiv(n * n_components, blockSize_dimN);

  // Do gradient updates
  float momentum = pre_momentum;
  float Z;

  if (verbose) printf("[Info] Start gradient updates!\n");
  for (int iter = 0; iter < max_iter; iter++) {
    if (iter == exaggeration_iter) {
      momentum = post_momentum;
      // Divide perplexities
      const float div = 1.0f / early_exaggeration;
      MLCommon::LinAlg::scalarMultiply(VAL, (const float *)VAL, div, NNZ,
                                       stream);
    }
    // Get norm(Y)
    MLCommon::LinAlg::rowNorm(norm, Y, n_components, n,
                              MLCommon::LinAlg::L2Norm, false, stream);
    //TSNE::get_norm_fast(Y, norm, n, n_components, stream, gridSize_N, blockSize_N);

    // Fast compute attractive forces from COO matrix
    TSNE::attractive_fast(VAL, COL, ROW, Y, norm, attract, NNZ, n, n_components,
                          stream, gridSize_NNZ, blockSize_NNZ);

    // Fast compute repulsive forces
    Z = TSNE::repulsive_fast(Y, repel, norm, Q_sum, n, n_components, stream);
    if (verbose && iter % 100 == 0)
      printf("[Info]  Z at iter = %d is %lf.\n", iter, Z);

    // Integrate forces with momentum
    TSNE::apply_forces(attract, means, repel, Y, iY, gains, n, n_components, Z,
                       min_gain, momentum, eta, stream, gridSize_dimN,
                       blockSize_dimN, gains_add, gains_mult);

    if (momentum > 0.8) momentum -= 0.005;
  }

  printf("[Info]  TSNE has finished!\n");
  // Clean up
  P_PT.destroy();

  d_alloc->deallocate(norm, sizeof(float) * n, stream);
  d_alloc->deallocate(Q_sum, sizeof(float) * n, stream);
  d_alloc->deallocate(sum, sizeof(double), stream);

  d_alloc->deallocate(attract, sizeof(float) * n * n_components, stream);
  d_alloc->deallocate(repel, sizeof(float) * n * n_components, stream);

  d_alloc->deallocate(iY, sizeof(float) * n * n_components, stream);
  d_alloc->deallocate(gains, sizeof(float) * n * n_components, stream);
  d_alloc->deallocate(means, sizeof(float) * n_components, stream);
}

}  // namespace ML
