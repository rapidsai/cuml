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
#include <cuml/manifold/tsne.h>
#include "../../src_prims/utils.h"
#include "distances.h"
#include "exact_kernels.h"
#include "utils.h"

#include "barnes_hut.h"
#include "exact_tsne.h"

#include <cuml/decomposition/pca.hpp>
#include <linalg/transpose.h>

#define device_buffer MLCommon::device_buffer

namespace ML {

/**
 * @brief Dimensionality reduction via TSNE using either Barnes Hut O(NlogN) or brute force O(N^2).
 * @input param handle: The GPU handle.
 * @input param X: The dataset you want to apply TSNE on.
 * @output param embedding: The final embedding. Will overwrite this internally.
 * @input param n: Number of rows in data X.
 * @input param p: Number of columns in data X.
 * @input param dim: Number of output dimensions for embeddings.
 * @input param n_neighbors: Number of nearest neighbors used.
 * @input param theta: Float between 0 and 1. Tradeoff for speed (0) vs accuracy (1) for Barnes Hut only.
 * @input param epssq: A tiny jitter to promote numerical stability.
 * @input param perplexity: How many nearest neighbors are used during the construction of Pij.
 * @input param perplexity_max_iter: Number of iterations used to construct Pij.
 * @input param perplexity_tol: The small tolerance used for Pij to ensure numerical stability.
 * @input param early_exaggeration: How much early pressure you want the clusters in TSNE to spread out more.
 * @input param exaggeration_iter: How many iterations you want the early pressure to run for.
 * @input param min_gain: Rounds up small gradient updates.
 * @input param pre_learning_rate: The learning rate during the exaggeration phase.
 * @input param post_learning_rate: The learning rate after the exaggeration phase.
 * @input param max_iter: The maximum number of iterations TSNE should run for.
 * @input param min_grad_norm: The smallest gradient norm TSNE should terminate on.
 * @input param pre_momentum: The momentum used during the exaggeration phase.
 * @input param post_momentum: The momentum used after the exaggeration phase.
 * @input param random_state: Set this to -1 for pure random intializations or >= 0 for reproducible outputs.
 * @input param verbose: Whether to print error messages or not.
 * @input param init: Intialization type using IntializationType enum
 * @input param barnes_hut: Whether to use the fast Barnes Hut or use the slower exact version.
 */
void TSNE_fit(const cumlHandle &handle, float *X, float *embedding,
              const int n, const int p, const int dim, int n_neighbors,
              const float theta, const float epssq, float perplexity,
              const int perplexity_max_iter, const float perplexity_tol,
              const float early_exaggeration, const int exaggeration_iter,
              const float min_gain, const float pre_learning_rate,
              const float post_learning_rate, const int max_iter,
              const float min_grad_norm, const float pre_momentum,
              const float post_momentum, const long long random_state,
              const bool verbose, const IntializationType init, bool barnes_hut)
{
  ASSERT(n > 0 && p > 0 && dim > 0 && n_neighbors > 0 && X != NULL &&
         embedding != NULL, "Wrong input args");

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

  //---------------------------------------------------
  // PCA Intialization via Divide n Conquer Eigendecomposition
  float *A;
  device_buffer<float> X_C_contiguous(d_alloc, stream);

  reset_timers();
  START_TIMER;
  if (init == PCA_Intialization) {
    if (verbose) printf("[Info] Now performing PCA Intialization!\n");

    paramsPCA params;
    params.n_components = dim;
    params.n_rows = n;
    params.n_cols = p;
    params.whiten = false;
    params.n_iterations = 15;
    params.tol = 1e-7;
    params.algorithm = COV_EIG_DQ;

    device_buffer<float> components(d_alloc, stream, p * dim);
    device_buffer<float> explained_var(d_alloc, stream, dim);
    device_buffer<float> explained_var_ratio(d_alloc, stream, dim);
    device_buffer<float> singular_vals(d_alloc, stream, dim);
    device_buffer<float> mu(d_alloc, stream, p);
    device_buffer<float> noise_vars(d_alloc, stream, 1);

    ML::pcaFitTransform((cumlHandle &)handle, X, embedding, components.data(),
                        explained_var.data(), explained_var_ratio.data(),
                        singular_vals.data(), mu.data(), noise_vars.data(),
                        params);

    // Scale components
    thrust::device_ptr<float> Y_ = thrust::device_pointer_cast(embedding);
    const float max = fabs(
      *thrust::max_element(thrust::cuda::par.on(stream), Y_, Y_ + n * dim));
    const float min = fabs(
      *thrust::min_element(thrust::cuda::par.on(stream), Y_, Y_ + n * dim));

    float total_maximum = (max > min) ? max : min;
    if (verbose)
    {
      printf("[Info] PCA largest value in intialization = %.3f\n",
             total_maximum);
    }
    if (total_maximum == 0) {
      // Intialize with random numbers since total_maximum == 0
      random_vector(embedding, -0.001f, 0.001f, n * dim, stream, random_state);
    }
    else {
      MLCommon::LinAlg::scalarMultiply(embedding, embedding,
                                       1.0f / total_maximum, n * dim, stream);
    }

    // Now transpose the data to make it C-Contiguous
    X_C_contiguous.resize(n * p, stream);
    MLCommon::LinAlg::transpose(X, X_C_contiguous.data(), n, p,
                                handle.getImpl().getCublasHandle(), stream);

    A = X_C_contiguous.data();

    // Immediately free the buffers
    components.release(stream);
    explained_var.release(stream);
    explained_var_ratio.release(stream);
    singular_vals.release(stream);
    mu.release(stream);
    noise_vars.release(stream);
  }
  else {
    A = X;
  }
  END_TIMER(PCATime);

  START_TIMER;
  //---------------------------------------------------
  // Get distances
  if (verbose) printf("[Info] Getting distances.\n");

  device_buffer<float> distances_(d_alloc, stream, n*n_neighbors);
  float *distances = distances_.data();
  device_buffer<long> indices_(d_alloc, stream, n*n_neighbors);
  long *temp_indices = indices_.data();
  TSNE::get_distances(A, n, p, temp_indices, distances, n_neighbors, d_alloc, stream);

  if (init == PCA_Intialization) {
    X_C_contiguous.release(stream);  // remove C contiguous layout
  }

  //---------------------------------------------------
  END_TIMER(DistancesTime);

  START_TIMER;
  //---------------------------------------------------
  // Normalize distances
  if (verbose) {
    printf("[Info] Now normalizing distances so exp(D) doesn't explode.\n");
  }
  TSNE::normalize_distances(n, distances, n_neighbors, stream);
  //---------------------------------------------------
  END_TIMER(NormalizeTime);

  START_TIMER;
  //---------------------------------------------------
  // Optimal perplexity
  if (verbose) {
    printf("[Info] Searching for optimal perplexity via bisection search.\n");
  }

  size_t workspace_size = 0;
  const int NNZ = (2 * n * n_neighbors);

  device_buffer<float> P_(d_alloc, stream, NNZ);
  workspace_size += n*n_neighbors*sizeof(float);

  float *P = P_.data();
  TSNE::perplexity_search(distances, P, perplexity, perplexity_max_iter,
                          perplexity_tol, n, n_neighbors, handle);

  distances_.release(stream);
  //---------------------------------------------------
  END_TIMER(PerplexityTime);


  START_TIMER;
  //---------------------------------------------------  

  float *VAL = P;
  device_buffer<int> COL_(d_alloc, stream, NNZ);
  int *COL = COL_.data();

  /*
  Reuse temp_indices[long](n*n_neighbors) as ROW indices for the COO P+P.T matrix.
  Normally sizeof(long) = 2*sizeof(int), so we can reuse it!
  */
  int *ROW = (int*)temp_indices;
  device_buffer<int> ROW_(d_alloc, stream);

  if (sizeof(long) < 2*sizeof(int)) {
    ROW_.resize(NNZ, stream);
    ROW = ROW_.data();
  }
  else {
    workspace_size += NNZ * sizeof(int);
  }

  int *row_sizes = NULL;
  if ((sizeof(float)*n*dim >= sizeof(int)*n*2) and (init == Random_Intialization)) {
    row_sizes = (int*) embedding;
    workspace_size += 2*n*sizeof(int);
  }

  TSNE::symmetrize_perplexity(P, temp_indices, n, n_neighbors,
                              early_exaggeration, VAL, COL, ROW,
                              row_sizes, stream, handle);

  if (sizeof(long) < 2*sizeof(int)) {
    ROW_.release(stream);
  }

  //---------------------------------------------------
  END_TIMER(SymmetrizeTime);

  if (barnes_hut) {
    TSNE::Barnes_Hut(VAL, COL, ROW, NNZ, handle, embedding, n, theta, epssq,
                     early_exaggeration, exaggeration_iter, min_gain,
                     pre_learning_rate, post_learning_rate, max_iter,
                     min_grad_norm, pre_momentum, post_momentum, random_state,
                     verbose, init, workspace_size);
  }
  else {
    TSNE::Exact_TSNE(VAL, COL, ROW, NNZ, handle, embedding, n, dim,
                     early_exaggeration, exaggeration_iter, min_gain,
                     pre_learning_rate, post_learning_rate, max_iter,
                     min_grad_norm, pre_momentum, post_momentum, random_state,
                     verbose, init, workspace_size);
  }

  if (verbose) printf("[Info] TSNE has completed!\n");
}


}  // namespace ML
#undef device_buffer
