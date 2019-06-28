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

#include "distances.h"
#include "exact_kernels.h"
#include "symmetrize.h"
#include "tsne/tsne.h"
#include "utils.h"

#include "barnes_hut.h"
#include "exact_tsne.h"

namespace ML {

void TSNE_fit(const cumlHandle &handle, const float *X, float *Y, const int n,
              const int p, const int dim = 2, int n_neighbors = 1023,
              const float theta = 0.5f, const float epssq = 0.0025,
              float perplexity = 50.0f, const int perplexity_max_iter = 100,
              const float perplexity_tol = 1e-5,
              const float early_exaggeration = 12.0f,
              const int exaggeration_iter = 250, const float min_gain = 0.01f,
              const float pre_learning_rate = 200.0f,
              const float post_learning_rate = 500.0f,
              const int max_iter = 1000, const float min_grad_norm = 1e-7,
              const float pre_momentum = 0.5, const float post_momentum = 0.8,
              const long long random_state = -1, const bool verbose = true,
              const bool intialize_embeddings = true, bool barnes_hut = true) {
  assert(n > 0 && p > 0 && dim > 0 && n_neighbors > 0 && X != NULL &&
         Y != NULL);
  if (dim > 2 and barnes_hut) {
    barnes_hut = false;
    printf(
      "[Warn]  Barnes Hut only works for dim == 2. Switching to exact "
      "solution.\n");
  }
  if (n_neighbors > n) n_neighbors = n;
  if (n_neighbors > 1023) {
    printf("[Warn]  FAISS only supports maximum n_neighbors = 1023.\n");
    n_neighbors = 1023;
  }
  // Perplexity must be less than number of datapoints
  // "How to Use t-SNE Effectively" https://distill.pub/2016/misread-tsne/
  if (perplexity > n) perplexity = n;

  if (verbose) {
    printf("[Info]  Data size = (%d, %d) with dim = %d perplexity = %f\n", n, p,
           dim, perplexity);
    if (perplexity < 5 or perplexity > 50)
      printf(
        "[Warn]  Perplexity should be within ranges (5, 50). Your results "
        "might be a bit strange...\n");
    if (n_neighbors < perplexity * 3.0f)
      printf(
        "[Warn]  # of Nearest Neighbors should be at least 3 * perplexity. "
        "Your results might be a bit strange...\n");
  }

  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  START_TIMER;
  //---------------------------------------------------
  // Get distances
  if (verbose) printf("[Info] Getting distances.\n");
  float *distances =
    (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);
  long *indices =
    (long *)d_alloc->allocate(sizeof(long) * n * n_neighbors, stream);
  TSNE::get_distances(X, n, p, indices, distances, n_neighbors, stream);
  //---------------------------------------------------
  END_TIMER(DistancesTime);

  START_TIMER;
  //---------------------------------------------------
  // Normalize distances
  if (verbose)
    printf("[Info] Now normalizing distances so exp(D) doesn't explode.\n");
  TSNE::normalize_distances(n, distances, n_neighbors, stream);
  //---------------------------------------------------
  END_TIMER(NormalizeTime);

  START_TIMER;
  //---------------------------------------------------
  // Optimal perplexity
  if (verbose)
    printf("[Info] Searching for optimal perplexity via bisection search.\n");
  float *P =
    (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);
  const float P_sum =
    TSNE::perplexity_search(distances, P, perplexity, perplexity_max_iter,
                            perplexity_tol, n, n_neighbors, stream);
  d_alloc->deallocate(distances, sizeof(float) * n * n_neighbors, stream);
  if (verbose) printf("[Info] Perplexity sum = %f\n", P_sum);
  //---------------------------------------------------
  END_TIMER(PerplexityTime);

  START_TIMER;
  //---------------------------------------------------
  // Convert data to COO layout
  struct COO_Matrix_t COO_Matrix = TSNE::symmetrize_perplexity(
    P, indices, n, n_neighbors, P_sum, early_exaggeration, stream, handle);
  const int NNZ = COO_Matrix.NNZ;
  float *VAL = COO_Matrix.VAL;
  const int *COL = COO_Matrix.COL;
  const int *ROW = COO_Matrix.ROW;
  //---------------------------------------------------
  END_TIMER(SymmetrizeTime);

  if (barnes_hut) {
    TSNE::Barnes_Hut(VAL, COL, ROW, NNZ, handle, Y, n, theta, epssq,
                     early_exaggeration, exaggeration_iter, min_gain,
                     pre_learning_rate, post_learning_rate, max_iter,
                     min_grad_norm, pre_momentum, post_momentum, random_state,
                     verbose);
  } else {
    TSNE::Exact_TSNE(VAL, COL, ROW, NNZ, handle, Y, n, dim, early_exaggeration,
                     exaggeration_iter, min_gain, pre_learning_rate,
                     post_learning_rate, max_iter, min_grad_norm, pre_momentum,
                     post_momentum, random_state, verbose,
                     intialize_embeddings);
  }

  d_alloc->deallocate(COO_Matrix.VAL, sizeof(float) * NNZ, stream);
  d_alloc->deallocate(COO_Matrix.COL, sizeof(int) * NNZ, stream);
  d_alloc->deallocate(COO_Matrix.ROW, sizeof(int) * NNZ, stream);
}

}  // namespace ML
