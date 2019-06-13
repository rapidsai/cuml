
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
          const int p, const int n_components, const int n_neighbors,
          const float perplexity, const int perplexity_epochs,
          const int perplexity_tol, const float early_exaggeration,
          const int exaggeration_iter, const float min_gain, const float eta,
          const int epochs, const float pre_momentum, const float post_momentum,
          const long long seed) {
  auto d_alloc = handle.getDeviceAllocator();

  cudaStream_t stream = handle.getStream();

  // Some preliminary intializations for cuBLAS and cuML
  DEBUG("[Info]	Create cuBLAS and cuML handles.\n");
  const int k = n_components;

  cublasHandle_t blas_handle = handle.getImpl().getCublasHandle();

  const float neg2 = -2.0f, beta = 0.0f, one = 1.0f;

  random_vector(Y, -0.05f, 0.05f, n * k, seed, false, stream);

  // Get distances
  DEBUG("[Info]	Get distances\n");
  float *distances =
    (float *)d_alloc->allocate(n * n_neighbors * sizeof(float), stream);
  long *indices =
    (long *)d_alloc->allocate(sizeof(long) * n * n_neighbors, stream);
  get_distances(X, n, p, indices, distances, n_neighbors, stream);

  // Get perplexity
  DEBUG("[Info]	Get perplexity\n");
  float *P = (float *)d_alloc->allocate(
    sizeof(float) * n * n_neighbors,
    stream);  //cmalloc(sizeof(float)*n*n_neighbors, false);
  float P_sum = determine_sigmas(distances, P, perplexity, perplexity_epochs,
                                 perplexity_tol, n, n_neighbors);
  d_alloc->deallocate(distances, n * n_neighbors * sizeof(float), stream);
  DEBUG("[Info]	P_sum = %f\n", P_sum);

  // Convert data to COO layout
  DEBUG("[Info]	Convert to COO layout and symmetrize\n");
  COO_t<float> P_PT;
  symmetrize_perplexity(P, indices, &P_PT, n, n_neighbors, P_sum,
                        early_exaggeration, stream);

  // Allocate data [NOTICE all Fortran Contiguous]
  DEBUG("[Info]	Malloc data and space\n");
  float *noise = (float *)d_alloc->allocate(sizeof(float) * n * n, stream);

  float *Q = (float *)d_alloc->allocate(sizeof(float) * n * n, stream);
  float *norm = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  float *Q_sum = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  float *sum = (float *)d_alloc->allocate(sizeof(float), stream);

  float *attract = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
  float *repel = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);

  float *iY = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
  float *gains = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);

  // Do gradient updates
  float momentum = pre_momentum;
  float Z;
  int error;

  DEBUG("[Info]	Start iterations\n");
  for (int iter = 0; iter < epochs; iter++) {
    if (iter == 100) momentum = post_momentum;

    if (iter == exaggeration_iter) {
      float div = 1.0f / early_exaggeration;
      inplace_multiply(P_PT.vals, P_PT.nnz, div);
    }

    // Get norm(Y)
    get_norm(Y, norm, n, k);
#if IF_DEBUG
    printf("[Info]	Norm(y)\n\n");
    std::cout << MLCommon::arr2Str(norm, 20, "norm", stream) << std::endl;
#endif

    // Do -2 * (Y @ Y.T)
    if (error = cublasSsyrk(blas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n,
                            k, &neg2, Y, n, &beta, Q, n)) {
      DEBUG("[ERROR]	Error from BLAS = %d", error);
      break;
    }
#if IF_DEBUG
    printf("[Info]	Y @ Y.T\n\n");
    std::cout << MLCommon::arr2Str(Q, 20, "YYT", stream) << std::endl;
#endif

    // Form T = 1 / (1+d)
    Z = form_t_distribution(Q, norm, n, Q_sum, sum);
    DEBUG("[Info]	Z =  %lf iter = %d\n", Z, iter);
#if IF_DEBUG
    printf("[Info]	Q 1/(1+d)\n\n");
    std::cout << MLCommon::arr2Str(Q, 20, "QQ", stream);

    printf("[Info]	Q_sum\n\n");
    std::cout << MLCommon::arr2Str(Q_sum, 20, "Q_sum", stream);

    // float sum_;
    // cudaMemcpy(&sum_, sum, sizeof(float), cudaMemcpyDeviceToHost);
    // printf("[Info]	sum again = %lf\n\n", sum_);
#endif

    // Compute attractive forces with COO matrix
    attractive_forces(P_PT.vals, P_PT.cols, P_PT.rows, Q, Y, attract, P_PT.nnz,
                      n, k);

    // Change Q to Q**2 for repulsion
    postprocess_Q(Q, Q_sum, n);

    // Compute repel_1 = Q @ Y
    if (error =
          cublasSsymm(blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, n,
                      k, &one, Q, n, Y, n, &beta, repel, n)) {
      DEBUG("[ERROR]	Error from BLAS = %d", error);
      break;
    }

    // Repel_2 = mean contributions yi - yj
    // Repel = Repel_1 - Repel_2
    repel_minus_QY(repel, Q_sum, Y, n, k);

    // Integrate forces with momentum
    apply_forces(attract, repel, Y, iY, noise, gains, n, k, Z, min_gain,
                 momentum, eta);

#if IF_DEBUG
    break;
#endif
  }

  P_PT.destroy();

  d_alloc->deallocate(noise, sizeof(float) * n * n, stream);

  d_alloc->deallocate(Q, sizeof(float) * n * n, stream);
  d_alloc->deallocate(norm, sizeof(float) * n, stream);
  d_alloc->deallocate(Q_sum, sizeof(float) * n, stream);
  d_alloc->deallocate(sum, sizeof(float), stream);

  d_alloc->deallocate(attract, sizeof(float) * n * k, stream);
  d_alloc->deallocate(repel, sizeof(float) * n * k, stream);

  d_alloc->deallocate(iY, sizeof(float) * n * k, stream);
  d_alloc->deallocate(gains, sizeof(float) * n * k, stream);
}

}  // namespace ML
