
#pragma once

#include <float.h>
#include <math.h>
#include "utils.h"
#define ceil(a, b) ((a + b - 1) / b)

namespace ML {

using namespace ML;
using namespace MLCommon;

/**
 * @TODO: This is just a unary element-wise operation. Use unary element-wise primitive
 */
__global__ void __inplace_multiply(float *__restrict__ X, const int n,
                                   const float mult) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) X[i] *= mult;
}

template <int TPB_X = 32>
inline void inplace_multiply(float *__restrict__ X, const int n,
                             const float mult, cudaStream_t stream) {
  int blks = MLCommon::ceildiv(n, TPB_X);
  dim3 grid(blks, 1, 1);
  dim3 blk(TPB_X, 1, 1);

  __inplace_multiply<<<grid, blk, 0, stream>>>(X, n, mult);
  CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void __determine_sigmas_row(const float *__restrict__ distances,
                                       float *__restrict__ P,
                                       const float perplexity,
                                       const float desired_entropy,
                                       float *__restrict__ P_sum,
                                       const int epochs, const float tol,
                                       const int n, const int k) {
  // For every item in row
  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) {
    float beta_min = -INFINITY;
    float beta_max = INFINITY;
    float beta = 1;
    float sum_Pi = 0;
    float sum_P_row, sum_disti_Pi, div_sum_Pi;
    float entropy, entropy_diff;
    register int ik = i * k;

    for (int step = 0; step < epochs; step++) {
      sum_Pi = FLT_EPSILON;
      sum_disti_Pi = 0;
      sum_P_row = 0;

      // Exponentiate to get guassian
      for (int j = 0; j < k; j++) {
        P[ik + j] = __expf(-distances[ik + j] * beta);
        sum_Pi += P[ik + j];
      }

      // Normalize
      div_sum_Pi = 1.0f / sum_Pi;
      for (int j = 0; j < k; j++) {
        P[ik + j] *= div_sum_Pi;  // P[i*k + j] / sum_Pi
        sum_disti_Pi += distances[ik + j] * P[ik + j];
        sum_P_row += P[ik + j];
      }

      entropy = __logf(sum_Pi) + beta * sum_disti_Pi;
      entropy_diff = entropy - desired_entropy;
      if (fabs(entropy_diff) <= tol) break;

      // Bisection search
      if (entropy_diff > 0) {
        beta_min = beta;
        if (isinf(beta_max))
          beta *= 2.0f;
        else
          beta = (beta + beta_max) * 0.5f;
      } else {
        beta_max = beta;
        if (isinf(beta_min))
          beta *= 0.5f;
        else
          beta = (beta + beta_min) * 0.5f;
      }
    }
    atomicAdd(P_sum, sum_P_row);
  }
}

float determine_sigmas(const float *__restrict__ distances,
                       float *__restrict__ P, const float perplexity,
                       const int epochs, const float tol, const int n,
                       const int k, cudaStream_t stream) {
  const float desired_entropy = logf(perplexity);
  float *P_sum_, P_sum;
  cudaMalloc(&P_sum_, sizeof(float));
  cudaMemset(P_sum_, 0, sizeof(float));

  __determine_sigmas_row<<<ceil(n, 1024), 1024, 0, stream>>>(
    distances, P, perplexity, desired_entropy, P_sum_, epochs, tol, n, k);

#if IF_DEBUG
  // Check if all perplexities are 0.2
  float *check = (float*) malloc(sizeof(float)*n*k);
  cudaMemcpy(check, P, sizeof(float)*n*k, cudaMemcpyDeviceToHost);
  int yes = 0;
  for (int i = 0; i < sizeof(float)*n*k; i++)
    if (check[i] == 0.2f) yes++;
  printf("[IMPORTANT CHECK if (n*k = yes)]  n*k = %d   yes = %d\n\n", n*k, yes);
  assert(n*k == yes);
  free(check);
#endif

  cudaMemcpy(&P_sum, P_sum_, sizeof(float), cudaMemcpyDeviceToHost);
  return P_sum;
}

/**
 * @TODO: We have prims for this. Check out the linalg/norm.h prims
 */
__global__ void __get_norm(const float *__restrict__ Y,
                           float *__restrict__ norm, const int n, const int K) {
  const int i =
    (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in col
  const int j = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every col
  if (i < n && j < K)
    // norm[i] += Y[i, j]**2
    atomicAdd(&norm[i], Y[j * n + i] * Y[j * n + i]);
}

template <int TPB_X = 32, int TPB_Y = 32>
void get_norm(const float *__restrict__ Y, float *__restrict__ norm,
              const int n, const int K, cudaStream_t stream) {
  // Notice Y is F-Contiguous
  cudaMemset(norm, 0, sizeof(float) * n);
  static const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(K, threadsPerBlock.y));
  __get_norm<<<numBlocks, threadsPerBlock, 0, stream>>>(Y, norm, n, K);
}

/**
 * @TODO: We have a primitive for this. Thrust can also be used.
 */
__global__ void __sum_array(const float *__restrict__ X,
                            double *__restrict__ sum, const int n) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) atomicAdd(sum, (double)X[i]);
}

__global__ void __form_t_distribution(float *__restrict__ Q,
                                      const float *__restrict__ norm,
                                      const int n, float *__restrict__ sum_Q) {
  const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
  const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row

  if (i < n && j < n) {
    if (i == j)   Q[i*n + j] = 0.0f;
    else if (j > i) {
      Q[j*n + i] = Q[i*n + j] = q = 1.0f / (Q[i*n + j] + norm[i] + norm[j] + 1.0f);
      atomicAdd(&sum_Q[i], Q[i*n + j]);
    }
  }
}

template <int TPB_X = 32, int TPB_Y = 32>
double form_t_distribution(float *__restrict__ Q, const float *__restrict__ norm,
                          const int n, float *__restrict__ sum_Q,
                          double *__restrict__ sum, cudaStream_t stream) {
  cudaMemset(sum_Q, 0, sizeof(float) * n);

  static const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(n, threadsPerBlock.y));

  __form_t_distribution<<<numBlocks, threadsPerBlock, 0, stream>>>(Q, norm, n, sum_Q);
  CUDA_CHECK(cudaPeekAtLastError());

  double Z = (double) thrust::reduce(__STREAM__, sum_Q, sum_Q + n);
  return ((double)1.0f / (2.0f * Z));
}

__global__ void __attractive_forces(const float *__restrict__ VAL,
                                    const int *__restrict__ COL,
                                    const int *__restrict__ ROW,
                                    const float *__restrict__ Q,
                                    const float *__restrict__ Y,
                                    float *__restrict__ attract, const int NNZ,
                                    const int n, const int K) {
  // Notice attract, Y and repel are all F-contiguous
  const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < NNZ) {
    const int i = ROW[index];
    const int j = COL[index];
    const float PQ = VAL[index] * Q[i*n + j];
    for (int l = 0; l < K; l++)
      // attract[i*K + j] += PQ * (Y[i, j] - Y[j, j]);
      atomicAdd(&attract[l * n + i], PQ * (Y[l * n + i] - Y[l * n + j]));
  }
}
void attractive_forces(const float *__restrict__ VAL,
                       const int *__restrict__ COL, const int *__restrict__ ROW,
                       const float *__restrict__ Q, const float *__restrict__ Y,
                       float *__restrict__ attract, const int NNZ, const int n,
                       const int K, cudaStream_t stream) {
  cudaMemset(attract, 0, sizeof(float) * n * K);
  __attractive_forces<<<ceil(NNZ, 1024), 1024, 0, stream>>>(VAL, COL, ROW, Q, Y,
                                                            attract, NNZ, n, K);
}

__global__ void __postprocess_Q(float *__restrict__ Q,
                                float *__restrict__ sum_Q, const int n) {
  const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
  const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
  if (i < n && j < n) {
    Q[i * n + j] *= Q[i * n + j];
    atomicAdd(&sum_Q[i], Q[i * n + j]);
  }
}

template <int TPB_X = 32, int TPB_Y = 32>
void postprocess_Q(float *__restrict__ Q, float *__restrict__ sum_Q,
                   const int n, cudaStream_t stream) {
  cudaMemset(sum_Q, 0, sizeof(float) * n);
  const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(n, threadsPerBlock.y));
  __postprocess_Q<<<numBlocks, threadsPerBlock, 0, stream>>>(Q, sum_Q, n);
}

/**
 * @TODO: This is a unary elt-wise operation. Use the prims for this
 */
__global__ void __negative_array(float *__restrict__ X, const int n) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) X[i] *= -1;
}

__global__ void __repel_minus_QY(float *__restrict__ repel,
                                 const float *__restrict__ neg_sum_Q,
                                 const float *__restrict__ Y, const int n,
                                 const int K) {
  const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every column
  const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every item in column
  if (j < K && i < n)
    // repel[i*n + j] -= Q_sum[i] * Y[i*n + j];
    atomicAdd(&repel[j * n + i],
              neg_sum_Q[i] * Y[j * n + i]);  // Y, repel is F-Contiguous
}

template <int TPB_X = 32, int TPB_Y = 32>
void repel_minus_QY(float *__restrict__ repel, float *__restrict__ sum_Q,
                    const float *__restrict__ Y, const int n, const int K,
                    cudaStream_t stream) {
  int blks = MLCommon::ceildiv(n, TPB_X);

  dim3 grid(blks, 1, 1);
  dim3 blk(TPB_X, 1, 1);

  __negative_array<<<grid, blk, 0, stream>>>(sum_Q, n);

  static const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(ceil(K, threadsPerBlock.x), ceil(n, threadsPerBlock.y));

  __repel_minus_QY<<<numBlocks, threadsPerBlock, 0, stream>>>(repel, sum_Q, Y,
                                                              n, K);
}

__global__ void __apply_forces(const float *__restrict__ attract,
                               const float *__restrict__ repel,
                               float *__restrict__ Y, float *__restrict__ iY,
                               const float *__restrict__ noise,
                               float *__restrict__ gains, const int n,
                               const int K, const float Z, const float min_gain,
                               const float momentum, const float eta) {
  // Everything is F-Contiguous
  // NOTICE noise is a 1D array

  const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every column
  const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every item in column
  if (j < K && i < n) {
    const int index = j * n + i;
    // DY[:] = attract + Z*repel
    const float dy = attract[index] + Z * repel[index];
    // gains[:] = (gains + 0.2) * ((DY > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * ((DY > 0.) == (iY > 0.))
    if (signbit(dy) != signbit(iY[index]))
      gains[index] += 0.2f;
    else
      gains[index] *= 0.8f;
    // gains[gains < min_gain] = min_gain
    if (gains[index] < min_gain) gains[index] = min_gain;

    // iY[:] = momentum * iY - eta * (gains * DY)
    iY[index] = momentum * iY[index] - eta * (gains[index] * dy);

    // Y += iY + noise
    Y[index] += (iY[index] + noise[i]);
  }
}

template <int TPB_X = 32, int TPB_Y = 32>
void apply_forces(const float *__restrict__ attract,
                  const float *__restrict__ repel, float *__restrict__ Y,
                  float *__restrict__ iY, const float *__restrict__ noise,
                  float *__restrict__ gains, const int n, const int K,
                  const float Z, const float min_gain, const float momentum,
                  const float eta, cudaStream_t stream) {
  static const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(ceil(K, threadsPerBlock.x), ceil(n, threadsPerBlock.y));
  __apply_forces<<<numBlocks, threadsPerBlock, 0, stream>>>(
    attract, repel, Y, iY, noise, gains, n, K, Z, min_gain, momentum, eta);
}

}  // namespace ML
