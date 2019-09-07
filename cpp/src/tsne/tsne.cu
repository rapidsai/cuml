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
#include "../../src_prims/utils.h"
#include "distances.h"
#include "exact_kernels.h"
#include "tsne/tsne.h"
#include "unary_op.h"
#include "utils.h"

#include "barnes_hut.h"
#include "exact_tsne.h"
#include "spectral/spectral.h"

#define CHECK(x)                                                       \
  ASSERT(x == 0, "cuSolver or cuBLAS failed at line = %d file = %s\n", \
         __LINE__, __FILE__);

#define MIN(a, b) (a > b) ? b : a

namespace ML {

/**
 * @brief Dimensionality reduction via TSNE using either Barnes Hut O(NlogN) or brute force O(N^2).
 * @input param handle: The GPU handle.
 * @input param X: The dataset you want to apply TSNE on.
 * @output param embedding: The final embedding. Will overwrite this internally.
 * @input param n: Number of rows in data X.
 * @input param p: Number of columns in data X.
 * @input param dim: Number of output dimensions for embeddings embedding.
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
 * @input param spectral_intialization: Whether to intialize with spectral embedding. Acts like pseudo PCA.
 * @input param barnes_hut: Whether to use the fast Barnes Hut or use the slower exact version.
 */
void TSNE_fit(const cumlHandle &handle, const float *X, float *embedding,
              const int n, const int p, const int dim, int n_neighbors,
              const float theta, const float epssq, float perplexity,
              const int perplexity_max_iter, const float perplexity_tol,
              const float early_exaggeration, const int exaggeration_iter,
              const float min_gain, const float pre_learning_rate,
              const float post_learning_rate, const int max_iter,
              const float min_grad_norm, const float pre_momentum,
              const float post_momentum, const long long random_state,
              const bool verbose, const bool spectral_intialization,
              bool barnes_hut) {
  ASSERT(n > 0 && p > 0 && dim > 0 && n_neighbors > 0 && X != NULL &&
           embedding != NULL,
         "Wrong input args");

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
                            perplexity_tol, n, n_neighbors, handle);
  d_alloc->deallocate(distances, sizeof(float) * n * n_neighbors, stream);
  if (verbose) printf("[Info] Perplexity sum = %f\n", P_sum);
  //---------------------------------------------------
  END_TIMER(PerplexityTime);

  START_TIMER;
  //---------------------------------------------------
  // Convert data to COO layout
  MLCommon::Sparse::COO<float> COO_Matrix;
  TSNE::symmetrize_perplexity(P, indices, n, n_neighbors, P_sum,
                              early_exaggeration, &COO_Matrix, stream, handle);
  const int NNZ = COO_Matrix.nnz;
  float *VAL = COO_Matrix.vals;
  const int *COL = COO_Matrix.cols;
  const int *ROW = COO_Matrix.rows;
  //---------------------------------------------------
  END_TIMER(SymmetrizeTime);

  // Intialize via Sparse SVD for COO matrices
  int cols = n;
  int oversamples = 10;
  int k = MIN(2 + oversamples, cols);
  cusolverDnHandle_t cusolverH = NULL;
  CHECK(cusolverDnCreate(&cusolverH));
  cublasHandle_t cublasH = NULL;
  CHECK(cublasCreate(&cublasH));

  float *Y /*(n,k)*/ =
    (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
  float *Z /*(p,k)*/ =
    (float *)d_alloc->allocate(sizeof(float) * cols * k, stream);
  random_vector(Z, 0.0f, 1.0f, cols * k, stream, random_state,
                true);  // normal = true

  // Y, _ = np.linalg.qr(Y)
  int lwork_Y = 0;
  CHECK(cusolverDnSgeqrf_bufferSize(cusolverH, n, k, Y, n, &lwork_Y));
  float *work_Y = (float *)d_alloc->allocate(sizeof(float) * lwork_Y, stream);

  // Z, _ = np.linalg.qr(Z)
  int lwork_Z = 0;
  CHECK(cusolverDnSgeqrf_bufferSize(cusolverH, cols, k, Z, cols, &lwork_Z));
  float *work_Z = (float *)d_alloc->allocate(sizeof(float) * lwork_Z, stream);

  // Tau for both QR factorizations
  float *tau = (float *)d_alloc->allocate(sizeof(float) * k, stream);
  int *info = (int *)d_alloc->allocate(sizeof(int), stream);

  // Y = X @ Z
  MLCommon::Sparse::coo_gemm(&COO_Matrix, Z, k, Y, stream,
                             false);  // trans = false

  for (int i = 0; i < 3; i++) {
    // Y, _ = np.linalg.qr(Y)
    CHECK(cusolverDnSgeqrf(cusolverH, n, k, Y, n, tau, work_Y, lwork_Y, info));
    CHECK(
      cusolverDnSorgqr(cusolverH, n, k, k, Y, n, tau, work_Y, lwork_Y, info));

    // Z = X.T @ Y
    MLCommon::Sparse::coo_gemm(&COO_Matrix, Y, k, Z, stream,
                               true);  // trans = true

    // Z, _ = np.linalg.qr(Z)
    CHECK(cusolverDnSgeqrf(cusolverH, cols, k, Z, cols, tau, work_Z, lwork_Z,
                           info));
    CHECK(cusolverDnSorgqr(cusolverH, cols, k, k, Z, cols, tau, work_Z, lwork_Z,
                           info));

    // Y = X @ Z
    MLCommon::Sparse::coo_gemm(&COO_Matrix, Z, k, Y, stream,
                               false);  // trans = false
  }

  // Y, _ = np.linalg.qr(Y)
  CHECK(cusolverDnSgeqrf(cusolverH, n, k, Y, n, tau, work_Y, lwork_Y, info));
  CHECK(cusolverDnSorgqr(cusolverH, n, k, k, Y, n, tau, work_Y, lwork_Y, info));

  // Z(p,k) = Y.T @ X (or (X.T @ Y).T)
  MLCommon::Sparse::coo_gemm(&COO_Matrix, Y, k, Z, stream,
                             true);  // trans = true

  // T(k,k) = Z @ Z.T (or (Z.T @ Z))
  float *T = (float *)d_alloc->allocate(sizeof(float) * k * k, stream);

  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSsyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, k, cols, &alpha, Z,
              cols, &beta, T, k);

  // W, Uhat = np.linalg.eigh(T)
  float *W = (float *)d_alloc->allocate(sizeof(float) * k, stream);
  float *Uhat = (float *)d_alloc->allocate(sizeof(float) * k * k, stream);

  int lwork_T = 0;
  CHECK(cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_UPPER, k, T, k, W,
                                    &lwork_T));
  float *work_T = (float *)d_alloc->allocate(sizeof(float) * lwork_T, stream);

  CHECK(cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                         CUBLAS_FILL_MODE_UPPER, k, T, k, W, work_T, lwork_T,
                         info));

  int info_cpu;
  CUDA_CHECK(cudaMemcpy(&info_cpu, info, sizeof(int), cudaMemcpyDeviceToHost));

  printf("Lwork_Y = %d, Lwork_Z = %d lwork_T = %d Info = %d\n", lwork_Y,
         lwork_Z, lwork_T, info_cpu);

  d_alloc->deallocate(work_T, sizeof(float) * lwork_T, stream);
  d_alloc->deallocate(Uhat, sizeof(float) * k * k, stream);
  d_alloc->deallocate(W, sizeof(float) * k, stream);
  d_alloc->deallocate(T, sizeof(float) * k * k, stream);

  d_alloc->deallocate(work_Z, sizeof(float) * lwork_Z, stream);
  d_alloc->deallocate(work_Y, sizeof(float) * lwork_Y, stream);
  d_alloc->deallocate(tau, sizeof(float) * k, stream);
  d_alloc->deallocate(info, sizeof(int), stream);

  d_alloc->deallocate(Y, sizeof(float) * n * k, stream);
  d_alloc->deallocate(Z, sizeof(float) * cols * k, stream);

  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);

  if (barnes_hut) {
    TSNE::Barnes_Hut(VAL, COL, ROW, NNZ, handle, embedding, n, theta, epssq,
                     early_exaggeration, exaggeration_iter, min_gain,
                     pre_learning_rate, post_learning_rate, max_iter,
                     min_grad_norm, pre_momentum, post_momentum, random_state,
                     verbose, spectral_intialization);
  } else {
    TSNE::Exact_TSNE(VAL, COL, ROW, NNZ, handle, embedding, n, dim,
                     early_exaggeration, exaggeration_iter, min_gain,
                     pre_learning_rate, post_learning_rate, max_iter,
                     min_grad_norm, pre_momentum, post_momentum, random_state,
                     verbose, spectral_intialization);
  }

  COO_Matrix.destroy();
}

}  // namespace ML
