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

#include "arima_helpers.cuh"
#include "batched_kalman.hpp"

#include <nvToolsExt.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>

#include <cuml/cuml.hpp>

#include "common/nvtx.hpp"
#include "cuda_utils.h"
#include "linalg/binary_op.h"
#include "linalg/cublas_wrappers.h"
#include "matrix/batched_matrix.hpp"
#include "timeSeries/jones_transform.h"
#include "utils.h"

using MLCommon::Matrix::b_gemm;
using MLCommon::Matrix::b_kron;
using MLCommon::Matrix::b_solve;
using BatchedMatrix = MLCommon::Matrix::BatchedMatrix<double>;

namespace ML {

void nvtx_range_push(std::string msg) { ML::PUSH_RANGE(msg.c_str()); }

void nvtx_range_pop() { ML::POP_RANGE(); }

//! Thread-local Matrix-Vector multiplication.
template <int r>
__device__ void Mv_l(double* A, double* v, double* out) {
  for (int i = 0; i < r; i++) {
    double sum = 0.0;
    for (int j = 0; j < r; j++) {
      sum += A[i + j * r] * v[j];
    }
    out[i] = sum;
  }
}

//! Thread-local Matrix-Matrix multiplication.
template <int r>
__device__ void MM_l(double* A, double* B, double* out) {
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < r; j++) {
      double sum = 0.0;
      for (int k = 0; k < r; k++) {
        sum += A[i + k * r] * B[k + j * r];
      }
      out[i + j * r] = sum;
    }
  }
}

/**
 * Kalman loop kernel. Each thread computes kalman filter for a single series
 * and stores relevant matrices in registers.
 *
 * @tparam     r          Dimension of the state vector
 * @param[in]  ys         Batched time series
 * @param[in]  nobs       Number of observation per series
 * @param[in]  T          Batched transition matrix.            (r x r)
 * @param[in]  Z          Batched "design" vector               (1 x r)
 * @param[in]  RRT        Batched R*R.T (R="selection" vector)  (r x r)
 * @param[in]  P          Batched P                             (r x r)
 * @param[in]  alpha      Batched state vector                  (r x 1)
 * @param[in]  batch_size Batch size
 * @param[out] vs         Batched residuals                     (nobs)
 * @param[out] Fs         Batched variance of prediction errors (nobs)    
 * @param[out] sum_logFs  Batched sum of the logs of Fs         (1)
 */
template <int r>
__global__ void batched_kalman_loop_kernel(const double* ys, int nobs,
                                           const double* T, const double* Z,
                                           const double* RRT, const double* P,
                                           const double* alpha, int batch_size,
                                           double* vs, double* Fs,
                                           double* sum_logFs) {
  double l_RRT[r * r];
  double l_T[r * r];
  double l_Z[r];
  double l_P[r * r];
  double l_alpha[r];
  double l_K[r];
  double l_tmpA[r * r];
  double l_tmpB[r * r];

  int bid = blockDim.x * blockIdx.x + threadIdx.x;

  if (bid < batch_size) {
    constexpr int r2 = r * r;

    // load GM into registers
    {
      int b_r_offset = bid * r;
      int b_r2_offset = bid * r2;
      for (int i = 0; i < r2; i++) {
        l_RRT[i] = RRT[b_r2_offset + i];
        l_T[i] = T[b_r2_offset + i];
        l_P[i] = P[b_r2_offset + i];
      }
      for (int i = 0; i < r; i++) {
        l_Z[i] = Z[b_r_offset + i];
        l_alpha[i] = alpha[b_r_offset + i];
      }
    }

    double b_sum_logFs = 0.0;
    const double* b_ys = ys + bid * nobs;
    double* b_vs = vs + bid * nobs;
    double* b_Fs = Fs + bid * nobs;

    for (int it = 0; it < nobs; it++) {
      // 1. & 2.
      b_vs[it] = b_ys[it] - l_alpha[0];
      double _Fs = l_P[0];
      b_Fs[it] = _Fs;
      b_sum_logFs += log(_Fs);

      // 3.
      // MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
      // tmpA = P*Z.T
      Mv_l<r>(l_P, l_Z, l_tmpA);
      // tmpB = T*tmpA
      Mv_l<r>(l_T, l_tmpA, l_tmpB);

      // K = 1/Fs[it] * tmpB
      double _1_Fs = 1.0 / _Fs;
      for (int i = 0; i < r; i++) {
        l_K[i] = _1_Fs * l_tmpB[i];
      }

      // 4.
      // alpha = T*alpha + K*vs[it];
      Mv_l<r>(l_T, l_alpha, l_tmpA);
      double vs_it = b_vs[it];
      for (int i = 0; i < r; i++) {
        l_alpha[i] = l_tmpA[i] + l_K[i] * vs_it;
      }

      // 5.
      // MatrixT L = T - K*Z;
      // tmpA = KZ
      // tmpA[0] = K[0]*Z[0]
      // tmpA[1] = K[1]*Z[0]
      // tmpA[2] = K[0]*Z[1]
      // tmpA[3] = K[1]*Z[1]
      // pytest [i % 3 for i in range(9)] -> 0 1 2 0 1 2 0 1 2
      // pytest [i//3 % 3 for i in range(9)] -> 0 0 0 1 1 1 2 2 2
      for (int tid = 0; tid < r * r; tid++) {
        l_tmpA[tid] = l_K[tid % r] * l_Z[(tid / r) % r];
      }

      // tmpA = T-tmpA
      for (int tid = 0; tid < r * r; tid++) {
        l_tmpA[tid] = l_T[tid] - l_tmpA[tid];
      }
      // note: L = tmpA

      // 6.
      // tmpB = tmpA.transpose()
      // tmpB[0] = tmpA[0]
      // tmpB[1] = tmpA[2]
      // tmpB[2] = tmpA[1]
      // tmpB[3] = tmpA[3]
      for (int tid = 0; tid < r; tid++) {
        for (int i = 0; i < r; i++) {
          l_tmpB[tid + i * r] = l_tmpA[tid * r + i];
        }
      }
      // note: L.T = tmpB

      // P = T * P * L.transpose() + R * R.transpose();
      // tmpA = P*L.T
      MM_l<r>(l_P, l_tmpB, l_tmpA);
      // tmpB = T*tmpA;
      MM_l<r>(l_T, l_tmpA, l_tmpB);
      // P = tmpB + RRT
      for (int tid = 0; tid < r * r; tid++) {
        l_P[tid] = l_tmpB[tid] + l_RRT[tid];
      }
    }
    sum_logFs[bid] = b_sum_logFs;
  }
}

/**
 * Kalman loop for large matrices (seasonality).
 *
 * @note: a single-kernel approach could work here and would be more
 *        performant with a large batch size and small seasonal period.
 *        This approach is more performant when the state vector size gets
 *        large.
 * @todo: proper benchmarks of both approaches
 *
 * @param[in]  d_ys         Batched time series
 * @param[in]  nobs         Number of observation per series
 * @param[in]  T            Batched transition matrix.            (r x r)
 * @param[in]  Z            Batched "design" vector               (1 x r)
 * @param[in]  RRT          Batched R*R' (R="selection" vector)   (r x r)
 * @param[in]  P            Batched P                             (r x r)
 * @param[in]  alpha        Batched state vector                  (r x 1)
 * @param[in]  r            Dimension of the state vector
 * @param[out] d_vs         Batched residuals                     (nobs)
 * @param[out] d_Fs         Batched variance of prediction errors (nobs)    
 * @param[out] d_sum_logFs  Batched sum of the logs of Fs         (1)
 */
void _batched_kalman_loop_large_matrices(
  const double* d_ys, int nobs, const BatchedMatrix& T, const BatchedMatrix& Z,
  const BatchedMatrix& RRT, BatchedMatrix& P, BatchedMatrix& alpha, int r,
  double* d_vs, double* d_Fs, double* d_sum_logFs) {
  auto stream = T.stream();
  auto allocator = T.allocator();
  auto cublasHandle = T.cublasHandle();
  int nb = T.batches();
  int r2 = r * r;

  BatchedMatrix v_tmp1(r, 1, nb, cublasHandle, allocator, stream, false);
  BatchedMatrix v_tmp2(r, 1, nb, cublasHandle, allocator, stream, false);
  BatchedMatrix m_tmp1(r, r, nb, cublasHandle, allocator, stream, false);
  BatchedMatrix m_tmp2(r, r, nb, cublasHandle, allocator, stream, false);
  BatchedMatrix K(r, 1, nb, cublasHandle, allocator, stream, false);

  double* d_P = P.raw_data();
  double* d_alpha = alpha.raw_data();
  double* d_K = K.raw_data();
  double* d_v_tmp1 = v_tmp1.raw_data();
  double* d_v_tmp2 = v_tmp2.raw_data();

  auto counting = thrust::make_counting_iterator(0);
  for (int it = 0; it < nobs; it++) {
    // 1. & 2.
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       d_vs[bid * nobs + it] =
                         d_ys[bid * nobs + it] - d_alpha[bid * r];
                       double l_P = d_P[bid * r2];
                       d_Fs[bid * nobs + it] = l_P;
                       d_sum_logFs[bid] += log(l_P);
                     });

    ///TODO: optimize for Z = (1 0 ... 0)?

    // 3. K = 1/Fs[it] * T*P*Z'
    // v_tmp1 = P*Z'
    b_gemm(false, true, r, 1, r, 1.0, P, Z, 0.0, v_tmp1);
    // v_tmp2 = T*v_tmp1
    b_gemm(false, false, r, 1, r, 1.0, T, v_tmp1, 0.0, v_tmp2);
    // K = 1/Fs[it] * v_tmp2
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       double _1_Fs = 1.0 / d_Fs[bid * nobs + it];
                       for (int i = 0; i < r; i++) {
                         d_K[bid * r + i] = _1_Fs * d_v_tmp2[r * bid + i];
                       }
                     });

    // 4. alpha = T*alpha + K*vs[it]
    // v_tmp1 = T*alpha
    b_gemm(false, false, r, 1, r, 1.0, T, alpha, 0.0, v_tmp1);
    // alpha = v_tmp1 + K*vs[it]
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       double _vs = d_vs[bid * nobs + it];
                       for (int i = 0; i < r; i++) {
                         d_alpha[bid * r + i] =
                           d_v_tmp1[r * bid + i] + _vs * d_K[bid * r + i];
                       }
                     });

    // 5. L = T - K * Z
    // L = T (L is m_tmp1)
    MLCommon::copy(m_tmp1.raw_data(), T.raw_data(), nb * r2, stream);
    // L = - K * Z + L
    b_gemm(false, false, r, r, 1, -1.0, K, Z, 1.0, m_tmp1);

    // 6. P = T*P*L' + R*R'
    // m_tmp2 = P*L'
    b_gemm(false, true, r, r, r, 1.0, P, m_tmp1, 0.0, m_tmp2);
    // m_tmp1 = T*m_tmp2
    b_gemm(false, false, r, r, r, 1.0, T, m_tmp2, 0.0, m_tmp1);
    // P = m_tmp1 + R*R'
    MLCommon::LinAlg::binaryOp(
      d_P, m_tmp1.raw_data(), RRT.raw_data(), r2 * nb,
      [=] __device__(double a, double b) { return a + b; }, stream);
  }
}

/**
 * Wrapper around multiple functions that can execute the Kalman loop in
 * difference cases (for performance)
 */
void batched_kalman_loop(const double* ys, int nobs, const BatchedMatrix& T,
                         const BatchedMatrix& Z, const BatchedMatrix& RRT,
                         BatchedMatrix& P0, BatchedMatrix& alpha, int r,
                         double* vs, double* Fs, double* sum_logFs) {
  const int batch_size = T.batches();
  auto stream = T.stream();
  dim3 numThreadsPerBlock(32, 1);
  dim3 numBlocks(MLCommon::ceildiv<int>(batch_size, numThreadsPerBlock.x), 1);
  if (r <= 8) {
    switch (r) {
      case 1:
        batched_kalman_loop_kernel<1>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs);
        break;
      case 2:
        batched_kalman_loop_kernel<2>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs);
        break;
      case 3:
        batched_kalman_loop_kernel<3>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs);
        break;
      case 4:
        batched_kalman_loop_kernel<4>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs);
        break;
      case 5:
        batched_kalman_loop_kernel<5>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs);
        break;
      case 6:
        batched_kalman_loop_kernel<6>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs);
        break;
      case 7:
        batched_kalman_loop_kernel<7>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs);
        break;
      case 8:
        batched_kalman_loop_kernel<8>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
  } else {
    _batched_kalman_loop_large_matrices(ys, nobs, T, Z, RRT, P0, alpha, r, vs,
                                        Fs, sum_logFs);
  }
}  // namespace ML

template <int NUM_THREADS>
__global__ void batched_kalman_loglike_kernel(double* d_vs, double* d_Fs,
                                              double* d_sumLogFs, int nobs,
                                              int batch_size, double* sigma2,
                                              double* loglike) {
  using BlockReduce = cub::BlockReduce<double, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_threads = blockDim.x;
  double bid_sigma2 = 0.0;
  for (int it = 0; it < nobs; it += num_threads) {
    // vs and Fs are in time-major order (memory layout: column major)
    int idx = (it + tid) + bid * nobs;
    double d_vs2_Fs = 0.0;
    if (idx < nobs * batch_size) {
      d_vs2_Fs = d_vs[idx] * d_vs[idx] / d_Fs[idx];
    }
    __syncthreads();
    double partial_sum = BlockReduce(temp_storage).Sum(d_vs2_Fs, nobs - it);
    bid_sigma2 += partial_sum;
  }
  if (tid == 0) {
    bid_sigma2 /= nobs;
    sigma2[bid] = bid_sigma2;
    loglike[bid] = -.5 * (d_sumLogFs[bid] + nobs * log(bid_sigma2)) -
                   nobs / 2. * (log(2 * M_PI) + 1);
  }
}

void batched_kalman_loglike(double* d_vs, double* d_Fs, double* d_sumLogFs,
                            int nobs, int batch_size, double* sigma2,
                            double* loglike, cudaStream_t stream) {
  const int NUM_THREADS = 128;
  batched_kalman_loglike_kernel<NUM_THREADS>
    <<<batch_size, NUM_THREADS, 0, stream>>>(d_vs, d_Fs, d_sumLogFs, nobs,
                                             batch_size, sigma2, loglike);
  CUDA_CHECK(cudaGetLastError());
}

// Internal Kalman filter implementation that assumes data exists on GPU.
void _batched_kalman_filter(cumlHandle& handle, const double* d_ys, int nobs,
                            const BatchedMatrix& Zb, const BatchedMatrix& Tb,
                            const BatchedMatrix& Rb, int r, double* d_vs,
                            double* d_Fs, double* d_loglike, double* d_sigma2,
                            bool initP_kalman_it = false) {
  const size_t batch_size = Zb.batches();
  auto stream = handle.getStream();

  BatchedMatrix RRT = b_gemm(Rb, Rb, false, true);

  BatchedMatrix P(r, r, batch_size, handle.getImpl().getCublasHandle(),
                  handle.getDeviceAllocator(), stream, false);
  if (initP_kalman_it)
    // A single Kalman iteration
    P = b_gemm(Tb, Tb, false, true) -
        Tb * b_gemm(Zb, b_gemm(Zb, Tb, false, true), true, false) + RRT;
  else {
    // # (Durbin Koopman "Time Series Analysis" pg 138)
    // NumPy version
    //   invImTT = np.linalg.pinv(np.eye(r**2) - np.kron(T_bi, T_bi))
    //   P0 = np.reshape(invImTT @ (R_bi @ R_bi.T).ravel(), (r, r), order="F")
    ML::PUSH_RANGE("P0: (I-TxT)");
    BatchedMatrix I_m_TxT =
      BatchedMatrix::Identity(r * r, batch_size,
                              handle.getImpl().getCublasHandle(),
                              handle.getDeviceAllocator(), stream) -
      b_kron(Tb, Tb);
    ML::POP_RANGE();
    ML::PUSH_RANGE("(I-TxT)\\(R.R^T)");
    BatchedMatrix invI_m_TxT_x_RRTvec = b_solve(I_m_TxT, RRT.vec());
    ML::POP_RANGE();
    BatchedMatrix P0 = invI_m_TxT_x_RRTvec.mat(r, r);
    P = P0;
  }

  // init alpha to zero
  BatchedMatrix alpha(r, 1, batch_size, handle.getImpl().getCublasHandle(),
                      handle.getDeviceAllocator(), stream, true);

  // init vs, Fs
  // In batch-major format.
  double* d_sumlogFs;

  d_sumlogFs = (double*)handle.getDeviceAllocator()->allocate(
    sizeof(double) * batch_size, stream);

  // Reference implementation
  // For it = 1:nobs
  //  // 1.
  //   vs[it] = ys[it] - alpha(0,0);
  //  // 2.
  //   Fs[it] = P(0,0);

  //   if(Fs[it] < 0) {
  //     std::cout << "P=" << P << "\n";
  //     throw std::runtime_error("ERROR: F < 0");
  //   }
  //   3.
  //   MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
  //   4.
  //   alpha = T*alpha + K*vs[it];
  //   5.
  //   MatrixT L = T - K*Z;
  //   6.
  //   P = T * P * L.transpose() + R * R.transpose();
  //   loglikelihood += std::log(Fs[it]);
  // }

  batched_kalman_loop(d_ys, nobs, Tb, Zb, RRT, P, alpha, r, d_vs, d_Fs,
                      d_sumlogFs);

  // Finalize loglikelihood
  // 7. & 8.
  // double sigma2 = ((vs.array().pow(2.0)).array() / Fs.array()).mean();
  // double loglike = -.5 * (loglikelihood + nobs * std::log(sigma2));
  // loglike -= nobs / 2. * (std::log(2 * M_PI) + 1);

  batched_kalman_loglike(d_vs, d_Fs, d_sumlogFs, nobs, batch_size, d_sigma2,
                         d_loglike, stream);
  handle.getDeviceAllocator()->deallocate(d_sumlogFs,
                                          sizeof(double) * batch_size, stream);
}

static void init_batched_kalman_matrices(
  cumlHandle& handle, const double* d_ar, const double* d_ma,
  const double* d_sar, const double* d_sma, int nb, int p, int q, int P, int Q,
  int s, int r, double* d_Z_b, double* d_R_b, double* d_T_b) {
  ML::PUSH_RANGE(__func__);

  auto stream = handle.getStream();

  cudaMemsetAsync(d_Z_b, 0.0, r * nb * sizeof(double), stream);
  cudaMemsetAsync(d_R_b, 0.0, r * nb * sizeof(double), stream);
  cudaMemsetAsync(d_T_b, 0.0, r * r * nb * sizeof(double), stream);

  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + nb,
    [=] __device__(int bid) {
      // See TSA pg. 54 for Z,R,T matrices
      // Z = [1 0 0 0 ... 0]
      d_Z_b[bid * r] = 1.0;

      /*     |1.0        |
       * R = |theta_1    |
       *     | ...       |
       *     |theta_{r-1}|
       */
      d_R_b[bid * r] = 1.0;
      for (int i = 0; i < r - 1; i++) {
        d_R_b[bid * r + i + 1] =
          reduced_polynomial<false>(bid, d_ma, q, d_sma, Q, s, i + 1);
      }

      /*     |phi_1  1.0  0.0  ...  0.0|
       *     | .          1.0          |
       *     | .              .        |
       * T = | .                .   0.0|
       *     | .                  .    |
       *     | .                    1.0|
       *     |phi_r  0.0  0.0  ...  0.0|
       */
      double* batch_T = d_T_b + bid * r * r;
      for (int i = 0; i < r; i++) {
        batch_T[i] = reduced_polynomial<true>(bid, d_ar, p, d_sar, P, s, i + 1);
      }
      // shifted identity
      for (int i = 0; i < r - 1; i++) {
        batch_T[(i + 1) * r + i] = 1.0;
      }
    });
  ML::POP_RANGE();
}  // namespace ML

void batched_kalman_filter(cumlHandle& handle, const double* d_ys, int nobs,
                           const double* d_ar, const double* d_ma,
                           const double* d_sar, const double* d_sma, int p,
                           int q, int P, int Q, int s, int batch_size,
                           double* loglike, double* d_vs, bool host_loglike,
                           bool initP_kalman_it) {
  ML::PUSH_RANGE("batched_kalman_filter");

  const size_t ys_len = nobs;

  auto cublasHandle = handle.getImpl().getCublasHandle();
  auto stream = handle.getStream();
  auto allocator = handle.getDeviceAllocator();

  // see (3.18) in TSA by D&K
  int r = std::max(p + s * P, q + s * Q + 1);

  BatchedMatrix Zb(1, r, batch_size, cublasHandle, allocator, stream, false);
  BatchedMatrix Tb(r, r, batch_size, cublasHandle, allocator, stream, false);
  BatchedMatrix Rb(r, 1, batch_size, cublasHandle, allocator, stream, false);

  init_batched_kalman_matrices(handle, d_ar, d_ma, d_sar, d_sma, batch_size, p,
                               q, P, Q, s, r, Zb.raw_data(), Rb.raw_data(),
                               Tb.raw_data());

  ////////////////////////////////////////////////////////////
  // Computation

  double* d_Fs =
    (double*)allocator->allocate(ys_len * batch_size * sizeof(double), stream);
  double* d_sigma2 =
    (double*)allocator->allocate(batch_size * sizeof(double), stream);

  /* Create log-likelihood device array if host pointer is provided */
  double* d_loglike;
  if (host_loglike) {
    d_loglike =
      (double*)allocator->allocate(batch_size * sizeof(double), stream);
  } else {
    d_loglike = loglike;
  }

  _batched_kalman_filter(handle, d_ys, nobs, Zb, Tb, Rb, r, d_vs, d_Fs,
                         d_loglike, d_sigma2, initP_kalman_it);

  if (host_loglike) {
    /* Tranfer log-likelihood device -> host */
    MLCommon::updateHost(loglike, d_loglike, batch_size, stream);
    allocator->deallocate(d_loglike, batch_size * sizeof(double), stream);
  }

  allocator->deallocate(d_Fs, ys_len * batch_size * sizeof(double), stream);

  allocator->deallocate(d_sigma2, batch_size * sizeof(double), stream);

  ML::POP_RANGE();
}

/* AR and MA parameters have to be within a "triangle" region (i.e., subject to
 * an inequality) for the inverse transform to not return 'NaN' due to the
 * logarithm within the inverse. This function ensures that inequality is
 * satisfied for all parameters.
 */
void fix_ar_ma_invparams(const double* d_old_params, double* d_new_params,
                         int batch_size, int pq, cudaStream_t stream,
                         bool isAr = true) {
  CUDA_CHECK(cudaMemcpyAsync(d_new_params, d_old_params,
                             batch_size * pq * sizeof(double),
                             cudaMemcpyDeviceToDevice, stream));
  int n = pq;

  // The parameter must be within a "triangle" region. If not, we bring the
  // parameter inside by 1%.
  double eps = 0.99;
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size,
    [=] __device__(int ib) {
      for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
          sum += d_new_params[n - j - 1 + ib * n];
        }
        // AR is minus
        if (isAr) {
          // param < 1-sum(param)
          d_new_params[n - i - 1 + ib * n] =
            fmin((1 - sum) * eps, d_new_params[n - i - 1 + ib * n]);
          // param > -(1-sum(param))
          d_new_params[n - i - 1 + ib * n] =
            fmax(-(1 - sum) * eps, d_new_params[n - i - 1 + ib * n]);
        } else {
          // MA is plus
          // param < 1+sum(param)
          d_new_params[n - i - 1 + ib * n] =
            fmin((1 + sum) * eps, d_new_params[n - i - 1 + ib * n]);
          // param > -(1+sum(param))
          d_new_params[n - i - 1 + ib * n] =
            fmax(-(1 + sum) * eps, d_new_params[n - i - 1 + ib * n]);
        }
      }
    });
}

void batched_jones_transform(cumlHandle& handle, int p, int q, int P, int Q,
                             int intercept, int batch_size, bool isInv,
                             const double* h_params, double* h_Tparams) {
  int N = p + q + P + Q + intercept;
  auto alloc = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_params =
    (double*)alloc->allocate(N * batch_size * sizeof(double), stream);
  double* d_Tparams =
    (double*)alloc->allocate(N * batch_size * sizeof(double), stream);
  double *d_mu, *d_ar, *d_ma, *d_sar, *d_sma, *d_Tar, *d_Tma, *d_Tsar, *d_Tsma;
  allocate_params(alloc, stream, p, q, P, Q, batch_size, &d_ar, &d_ma, &d_sar,
                  &d_sma, intercept, &d_mu);
  allocate_params(alloc, stream, p, q, P, Q, batch_size, &d_Tar, &d_Tma,
                  &d_Tsar, &d_Tsma);

  MLCommon::updateDevice(d_params, h_params, N * batch_size, stream);

  unpack(d_params, d_mu, d_ar, d_ma, d_sar, d_sma, batch_size, p, q, P, Q,
         intercept, stream);

  batched_jones_transform(handle, p, q, P, Q, batch_size, isInv, d_ar, d_ma,
                          d_sar, d_sma, d_Tar, d_Tma, d_Tsar, d_Tsma);

  pack(batch_size, p, q, P, Q, intercept, d_mu, d_Tar, d_Tma, d_Tsar, d_Tsma,
       d_Tparams, stream);

  MLCommon::updateHost(h_Tparams, d_Tparams, N * batch_size, stream);

  alloc->deallocate(d_params, N * batch_size * sizeof(double), stream);
  alloc->deallocate(d_Tparams, N * batch_size * sizeof(double), stream);
  deallocate_params(alloc, stream, p, q, P, Q, batch_size, d_ar, d_ma, d_sar,
                    d_sma, intercept, d_mu);
  deallocate_params(alloc, stream, p, q, P, Q, batch_size, d_Tar, d_Tma, d_Tsar,
                    d_Tsma);
}

/**
 * Auxiliary function of batched_jones_transform to remove redundancy.
 * Applies the transform to the given batched parameters.
 */
void _transform_helper(cumlHandle& handle, const double* d_param,
                       double* d_Tparam, int k, int batch_size, bool isInv,
                       bool isAr) {
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();

  // inverse transform will produce NaN if parameters are outside of a
  // "triangle" region
  double* d_param_fixed =
    (double*)allocator->allocate(sizeof(double) * batch_size * k, stream);
  if (isInv) {
    fix_ar_ma_invparams(d_param, d_param_fixed, batch_size, k, stream, isAr);
  } else {
    CUDA_CHECK(cudaMemcpyAsync(d_param_fixed, d_param,
                               sizeof(double) * batch_size * k,
                               cudaMemcpyDeviceToDevice, stream));
  }
  MLCommon::TimeSeries::jones_transform(d_param_fixed, batch_size, k, d_Tparam,
                                        isAr, isInv, allocator, stream);

  allocator->deallocate(d_param_fixed, sizeof(double) * batch_size * k, stream);
}

void batched_jones_transform(cumlHandle& handle, int p, int q, int P, int Q,
                             int batch_size, bool isInv, const double* d_ar,
                             const double* d_ma, const double* d_sar,
                             const double* d_sma, double* d_Tar, double* d_Tma,
                             double* d_Tsar, double* d_Tsma) {
  ML::PUSH_RANGE("batched_jones_transform");

  if (p) _transform_helper(handle, d_ar, d_Tar, p, batch_size, isInv, true);
  if (q) _transform_helper(handle, d_ma, d_Tma, q, batch_size, isInv, false);
  if (P) _transform_helper(handle, d_sar, d_Tsar, P, batch_size, isInv, true);
  if (Q) _transform_helper(handle, d_sma, d_Tsma, Q, batch_size, isInv, false);

  ///TODO: tranform SAR and SMA coefficients?!

  ML::POP_RANGE();
}

}  // namespace ML
