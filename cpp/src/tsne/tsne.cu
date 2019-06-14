
#pragma once

#include "common/cumlHandle.hpp"

#include "tsne/tsne.h"

#include "cublas_v2.h"
#include "distances.h"
#include "kernels.h"
#include "utils.h"

namespace ML {
using namespace MLCommon;

void TSNE(const cumlHandle &handle, const float *X, float *Y, const int n,
          const int p, const int n_components = 2, int n_neighbors = 90,
          const float *distances_vector = NULL, const long *indices_vector = NULL,
          const float perplexity = 30.0f, const int perplexity_epochs = 100,
          const int perplexity_tol = 1e-5,
          const float early_exaggeration = 12.0f,
          const int exaggeration_iter = 250, const float min_gain = 0.01f,
          const float eta = 500.0f, const int epochs = 150,
          const float pre_momentum = 0.8, const float post_momentum = 0.5,
          const long long seed = -1, const bool initialize_embeddings = false) {
  auto d_alloc = handle.getDeviceAllocator();

  cudaStream_t stream = handle.getStream();

  assert(n > 0 && p > 0 && n_components > 0 && n_neighbors > 0);
  if (n_neighbors > n) n_neighbors = n;

  // Some preliminary intializations for cuBLAS and cuML
  DEBUG("[Info] Create cuBLAS and cuML handles.\n");
  const int k = n_components;
  cublasHandle_t BLAS = handle.getImpl().getCublasHandle();

  const float neg2 = -2.0f, beta = 0.0f, one = 1.0f;



  // Get distances
  DEBUG("[Info] Get distances\n");
  float *distances =
      (float *)d_alloc->allocate(n * n_neighbors * sizeof(float), stream);
  long *indices =
    (long *)d_alloc->allocate(sizeof(long) * n * n_neighbors, stream);

  if (distances == NULL && indices == NULL) {
    get_distances(X, n, p, indices, distances, n_neighbors, stream);
  }
  else {
    MLCommon::updateDevice(distances, distances_vector, n * n_neighbors, stream);
    MLCommon::updateDevice(indices, indices_vector, n * n_neighbors, stream);
  }



  normalize_distances(n, distances, n_neighbors, stream);
#if IF_DEBUG
    printf("[Info]  Normalized distances\n\n");
    std::cout << MLCommon::arr2Str(distances, 20, "Distances", stream) << std::endl;
#endif


  // Get perplexity
  DEBUG("[Info] Get perplexity\n");
  float *P = (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);
  float P_sum = determine_sigmas(distances, P, perplexity, perplexity_epochs,
                                 perplexity_tol, n, n_neighbors, stream);
  d_alloc->deallocate(distances, n * n_neighbors * sizeof(float), stream);
  DEBUG("[Info] P_sum = %f\n", P_sum);
#if IF_DEBUG
    printf("[Info]  Perplexity results\n\n");
    std::cout << MLCommon::arr2Str(P, 20, "Perplexity", stream) << std::endl;
#endif



  // Convert data to COO layout
  DEBUG("[Info] Convert to COO layout and symmetrize\n");
  COO_t<float> P_PT;
  symmetrize_perplexity(P, indices, &P_PT, n, n_neighbors, P_sum,
                        early_exaggeration, stream);
  CUDA_CHECK(cudaPeekAtLastError());
#if IF_DEBUG
    printf("[Info]  Symmetrized Perplexity results\n\n");
    std::cout << MLCommon::arr2Str(P_PT.vals, 20, "Perplexity", stream) << std::endl;

    printf("[Info]  COL\n\n");
    std::cout << MLCommon::arr2Str(P_PT.cols, 20, "Perplexity", stream) << std::endl;

    printf("[Info]  RWW\n\n");
    std::cout << MLCommon::arr2Str(P_PT.rows, 20, "Perplexity", stream) << std::endl;
#endif



  // Allocate data [NOTICE all Fortran Contiguous]
  DEBUG("[Info] Malloc data and space\n");
  float *noise = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  cudaMemset(noise, 0, sizeof(float) * n);
  //random_vector(noise, -0.003f, 0.003f, n, stream, seed);
  CUDA_CHECK(cudaPeekAtLastError());

  if (initialize_embeddings) {
    random_vector(Y, -0.1f, 0.1f, n * k, stream, seed);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  float *Q = (float *)d_alloc->allocate(sizeof(float) * n * n, stream);
  float *norm = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  float *Q_sum = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  double *sum = (double *)d_alloc->allocate(sizeof(double), stream);

  float *attract = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
  float *repel = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);

  float *iY = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
  float *gains = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);

  // Do gradient updates
  float momentum = pre_momentum;
  float Z;
  int error;

  DEBUG("[Info] Start iterations\n");
  for (int iter = 0; iter < epochs; iter++) {

    if (iter == 100) momentum = post_momentum;

    if (iter == exaggeration_iter) {
      float div = 1.0f / early_exaggeration;
      inplace_multiply(P_PT.vals, P_PT.nnz, div, stream);
    }

    // Get norm(Y)
    get_norm(Y, norm, n, k, stream);
#if IF_DEBUG
    printf("[Info]  Norm(y)\n\n");
    std::cout << MLCommon::arr2Str(norm, 20, "norm", stream) << std::endl;
#endif

    // Do -2 * (Y @ Y.T)
    if (error = cublasSsyrk(BLAS, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k,
                            &neg2, Y, n, &beta, Q, n)) {
      DEBUG("[ERROR]  Error from BLAS = %d", error);
      break;
    }
#if IF_DEBUG
    printf("[Info]  -2Y @ Y.T\n\n");
    std::cout << MLCommon::arr2Str(Q, 20, "-2YYT", stream) << std::endl;
#endif

    // Form T = 1 / (1+d)
    Z = form_t_distribution(Q, norm, n, Q_sum, sum, stream);
    CUDA_CHECK(cudaPeekAtLastError());

    DEBUG("[Info] Z =  %llf iter = %d\n", Z, iter);
#if IF_DEBUG
    printf("[Info]  Q 1/(1+d)\n\n");
    std::cout << MLCommon::arr2Str(Q, 20, "QQ", stream);

    printf("[Info]  Q_sum\n\n");
    std::cout << MLCommon::arr2Str(Q_sum, 20, "Q_sum", stream);
#endif

    // Compute attractive forces with COO matrix
    attractive_forces(P_PT.vals, P_PT.cols, P_PT.rows, Q, Y, attract, P_PT.nnz,
                      n, k, stream);
    CUDA_CHECK(cudaPeekAtLastError());
#if IF_DEBUG
    printf("[Info]  Attractive forces\n\n");
    std::cout << MLCommon::arr2Str(attract, 20, "attract", stream);
#endif


    // Change Q to Q**2 for repulsion
    postprocess_Q(Q, Q_sum, n, stream);
#if IF_DEBUG
    printf("[Info]  Q**2\n\n");
    std::cout << MLCommon::arr2Str(Q, 20, "Q**2", stream);

    printf("[Info]  Q_sum\n\n");
    std::cout << MLCommon::arr2Str(Q_sum, 20, "Q_sum", stream);
#endif


    // Compute repel_1 = Q @ Y
    if (error = cublasSsymm(BLAS, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, n,
                            k, &one, Q, n, Y, n, &beta, repel, n)) {
      DEBUG("[ERROR]  Error from BLAS = %d", error);
      break;
    }
    CUDA_CHECK(cudaPeekAtLastError());
#if IF_DEBUG
    printf("[Info]  Q @ Y\n\n");
    std::cout << MLCommon::arr2Str(repel, 20, "Q @ Y", stream);
#endif


    // Repel_2 = mean contributions yi - yj
    // Repel = Repel_1 - Repel_2
    repel_minus_QY(repel, Q_sum, Y, n, k, stream);
    CUDA_CHECK(cudaPeekAtLastError());
#if IF_DEBUG
    printf("[Info]  repel - mean @ Y\n\n");
    std::cout << MLCommon::arr2Str(repel, 20, "repel - mean @ Y", stream);
#endif


    // Integrate forces with momentum
    apply_forces(attract, repel, Y, iY, noise, gains, n, k, Z, min_gain,
                 momentum, eta, stream);
    CUDA_CHECK(cudaPeekAtLastError());
#if IF_DEBUG
    printf("[Info]  Y after integration\n\n");
    std::cout << MLCommon::arr2Str(Y, 20, "Y", stream);
#endif


#if IF_DEBUG
  break;
#endif
  }

  P_PT.destroy();

  d_alloc->deallocate(noise, sizeof(float) * n, stream);

  d_alloc->deallocate(Q, sizeof(float) * n * n, stream);
  d_alloc->deallocate(norm, sizeof(float) * n, stream);
  d_alloc->deallocate(Q_sum, sizeof(float) * n, stream);
  d_alloc->deallocate(sum, sizeof(double), stream);

  d_alloc->deallocate(attract, sizeof(float) * n * k, stream);
  d_alloc->deallocate(repel, sizeof(float) * n * k, stream);

  d_alloc->deallocate(iY, sizeof(float) * n * k, stream);
  d_alloc->deallocate(gains, sizeof(float) * n * k, stream);
}
}  // namespace ML
