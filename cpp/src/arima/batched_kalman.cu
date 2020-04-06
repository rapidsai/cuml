/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <algorithm>
#include <vector>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>

#include <cuml/cuml.hpp>
#include <cuml/tsa/batched_kalman.hpp>

#include "common/cumlHandle.hpp"
#include "common/nvtx.hpp"
#include "cuda_utils.h"
#include "linalg/batched/matrix.h"
#include "linalg/binary_op.h"
#include "linalg/cublas_wrappers.h"
#include "sparse/batched/csr.h"
#include "timeSeries/arima_helpers.h"
#include "utils.h"

namespace ML {

//! Thread-local Matrix-Vector multiplication.
template <int r>
__device__ void Mv_l(const double* A, const double* v, double* out) {
  for (int i = 0; i < r; i++) {
    double sum = 0.0;
    for (int j = 0; j < r; j++) {
      sum += A[i + j * r] * v[j];
    }
    out[i] = sum;
  }
}

//! Thread-local Matrix-Matrix multiplication.
template <int r, bool aT = false, bool bT = false>
__device__ void MM_l(const double* A, const double* B, double* out) {
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < r; j++) {
      double sum = 0.0;
      for (int k = 0; k < r; k++) {
        double Aik = aT ? A[k + i * r] : A[i + k * r];
        double Bkj = bT ? B[j + k * r] : B[k + j * r];
        sum += Aik * Bkj;
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
 * @param[in]  fc_steps   Number of steps to forecast
 * @param[in]  d_fc       Array to store the forecast
 */
template <int r>
__global__ void batched_kalman_loop_kernel(const double* ys, int nobs,
                                           const double* T, const double* Z,
                                           const double* RRT, const double* P,
                                           const double* alpha, int batch_size,
                                           double* vs, double* Fs,
                                           double* sum_logFs, int fc_steps = 0,
                                           double* d_fc = nullptr) {
  constexpr int r2 = r * r;
  double l_RRT[r2];
  double l_T[r2];
  // double l_Z[r]; // note: will be used when introducing exogeneous var.
  double l_P[r2];
  double l_alpha[r];
  double l_K[r];
  double l_tmp[r2];
  double l_TP[r2];

  int bid = blockDim.x * blockIdx.x + threadIdx.x;

  if (bid < batch_size) {
    // Load global mem into registers
    {
      int b_r_offset = bid * r;
      int b_r2_offset = bid * r2;
      for (int i = 0; i < r2; i++) {
        l_RRT[i] = RRT[b_r2_offset + i];
        l_T[i] = T[b_r2_offset + i];
        l_P[i] = P[b_r2_offset + i];
      }
      for (int i = 0; i < r; i++) {
        // l_Z[i] = Z[b_r_offset + i];
        l_alpha[i] = alpha[b_r_offset + i];
      }
    }

    double b_sum_logFs = 0.0;
    const double* b_ys = ys + bid * nobs;
    double* b_vs = vs + bid * nobs;
    double* b_Fs = Fs + bid * nobs;

    for (int it = 0; it < nobs; it++) {
      // 1. & 2.
      double vs_it;
      double _Fs = l_P[0];
      vs_it = b_ys[it] - l_alpha[0];
      b_vs[it] = vs_it;
      b_Fs[it] = _Fs;
      b_sum_logFs += log(_Fs);

      // 3. K = 1/Fs[it] * T*P*Z'
      // TP = T*P
      MM_l<r>(l_T, l_P, l_TP);
      // K = 1/Fs[it] * TP*Z' ; optimized for Z = (1 0 ... 0)
      double _1_Fs = 1.0 / _Fs;
      for (int i = 0; i < r; i++) {
        l_K[i] = _1_Fs * l_TP[i];
      }

      // 4. alpha = T*alpha + K*vs[it]
      // tmp = T*alpha
      Mv_l<r>(l_T, l_alpha, l_tmp);
      // alpha = tmp + K*vs[it]
      for (int i = 0; i < r; i++) {
        l_alpha[i] = l_tmp[i] + l_K[i] * vs_it;
      }

      // 5. L = T - K * Z
      // L = T (L is tmp)
      for (int i = 0; i < r2; i++) {
        l_tmp[i] = l_T[i];
      }
      // L = L - K * Z ; optimized for Z = (1 0 ... 0):
      // substract K to the first column of L
      for (int i = 0; i < r; i++) {
        l_tmp[i] -= l_K[i];
      }

      // 6. P = T*P*L' + R*R'
      // P = TP*L'
      MM_l<r, false, true>(l_TP, l_tmp, l_P);
      // P = P + RRT
      for (int i = 0; i < r2; i++) {
        l_P[i] += l_RRT[i];
      }
    }
    sum_logFs[bid] = b_sum_logFs;

    // Forecast
    double* b_fc = fc_steps ? d_fc + bid * fc_steps : nullptr;
    for (int i = 0; i < fc_steps; i++) {
      b_fc[i] = l_alpha[0];

      // alpha = T*alpha
      Mv_l<r>(l_T, l_alpha, l_tmp);
      for (int i = 0; i < r; i++) {
        l_alpha[i] = l_tmp[i];
      }
    }
  }
}

/**
 * Kalman loop for large matrices (r > 8).
 *
 * @param[in]  d_ys         Batched time series
 * @param[in]  nobs         Number of observation per series
 * @param[in]  T            Batched transition matrix.            (r x r)
 * @param[in]  Z            Batched "design" vector               (1 x r)
 * @param[in]  RRT          Batched R*R' (R="selection" vector)   (r x r)
 * @param[in]  P            Batched P                             (r x r)
 * @param[in]  alpha        Batched state vector                  (r x 1)
 * @param[in]  T_sparse     Batched sparse matrix T               (r x r)
 * @param[in]  r            Dimension of the state vector
 * @param[out] d_vs         Batched residuals                     (nobs)
 * @param[out] d_Fs         Batched variance of prediction errors (nobs)    
 * @param[out] d_sum_logFs  Batched sum of the logs of Fs         (1)
 * @param[in]  fc_steps     Number of steps to forecast
 * @param[in]  d_fc         Array to store the forecast
 */
void _batched_kalman_loop_large(
  const double* d_ys, int nobs,
  const MLCommon::LinAlg::Batched::Matrix<double>& T,
  const MLCommon::LinAlg::Batched::Matrix<double>& Z,
  const MLCommon::LinAlg::Batched::Matrix<double>& RRT,
  MLCommon::LinAlg::Batched::Matrix<double>& P,
  MLCommon::LinAlg::Batched::Matrix<double>& alpha,
  MLCommon::Sparse::Batched::CSR<double>& T_sparse, int r, double* d_vs,
  double* d_Fs, double* d_sum_logFs, int fc_steps = 0, double* d_fc = nullptr) {
  auto stream = T.stream();
  auto allocator = T.allocator();
  auto cublasHandle = T.cublasHandle();
  int nb = T.batches();
  int r2 = r * r;

  // Temporary matrices and vectors
  MLCommon::LinAlg::Batched::Matrix<double> v_tmp(r, 1, nb, cublasHandle,
                                                  allocator, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> m_tmp(r, r, nb, cublasHandle,
                                                  allocator, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> K(r, 1, nb, cublasHandle, allocator,
                                              stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> TP(r, r, nb, cublasHandle,
                                               allocator, stream, false);

  // Shortcuts
  double* d_P = P.raw_data();
  double* d_alpha = alpha.raw_data();
  double* d_K = K.raw_data();
  double* d_TP = TP.raw_data();
  double* d_m_tmp = m_tmp.raw_data();
  double* d_v_tmp = v_tmp.raw_data();

  CUDA_CHECK(cudaMemsetAsync(d_sum_logFs, 0, sizeof(double) * nb, stream));

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

    // 3. K = 1/Fs[it] * T*P*Z'
    // TP = T*P (also used later)
    if (r <= 32)
      MLCommon::Sparse::Batched::b_spmm(1.0, T_sparse, P, 0.0, TP);
    else
      MLCommon::LinAlg::Batched::b_gemm(false, false, r, r, r, 1.0, T, P, 0.0,
                                        TP);
    // K = 1/Fs[it] * TP*Z' ; optimized for Z = (1 0 ... 0)
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       double _1_Fs = 1.0 / d_Fs[bid * nobs + it];
                       for (int i = 0; i < r; i++) {
                         d_K[bid * r + i] = _1_Fs * d_TP[bid * r2 + i];
                       }
                     });

    // 4. alpha = T*alpha + K*vs[it]
    // v_tmp = T*alpha
    MLCommon::Sparse::Batched::b_spmv(1.0, T_sparse, alpha, 0.0, v_tmp);
    // alpha = v_tmp + K*vs[it]
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       double _vs = d_vs[bid * nobs + it];
                       for (int i = 0; i < r; i++) {
                         d_alpha[bid * r + i] =
                           d_v_tmp[bid * r + i] + _vs * d_K[bid * r + i];
                       }
                     });

    // 5. L = T - K * Z
    // L = T (L is m_tmp)
    MLCommon::copy(m_tmp.raw_data(), T.raw_data(), nb * r2, stream);
    // L = L - K * Z ; optimized for Z = (1 0 ... 0):
    // substract K to the first column of L
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       for (int i = 0; i < r; i++) {
                         d_m_tmp[bid * r2 + i] -= d_K[bid * r + i];
                       }
                     });
    // MLCommon::LinAlg::Batched::b_gemm(false, false, r, r, 1, -1.0, K, Z, 1.0,
    //                                   m_tmp);  // generic

    // 6. P = T*P*L' + R*R'
    // P = TP*L'
    MLCommon::LinAlg::Batched::b_gemm(false, true, r, r, r, 1.0, TP, m_tmp, 0.0,
                                      P);
    // P = P + R*R'
    MLCommon::LinAlg::binaryOp(
      d_P, d_P, RRT.raw_data(), r2 * nb,
      [=] __device__(double a, double b) { return a + b; }, stream);
  }

  // Forecast
  for (int i = 0; i < fc_steps; i++) {
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + nb,
      [=] __device__(int bid) { d_fc[bid * fc_steps + i] = d_alpha[bid * r]; });

    MLCommon::Sparse::Batched::b_spmv(1.0, T_sparse, alpha, 0.0, v_tmp);
    MLCommon::copy(d_alpha, v_tmp.raw_data(), r * nb, stream);
  }
}

/**
 * Wrapper around multiple functions that can execute the Kalman loop in
 * difference cases (for performance)
 */
void batched_kalman_loop(cumlHandle& handle, const double* ys, int nobs,
                         const MLCommon::LinAlg::Batched::Matrix<double>& T,
                         const MLCommon::LinAlg::Batched::Matrix<double>& Z,
                         const MLCommon::LinAlg::Batched::Matrix<double>& RRT,
                         MLCommon::LinAlg::Batched::Matrix<double>& P0,
                         MLCommon::LinAlg::Batched::Matrix<double>& alpha,
                         std::vector<bool>& T_mask, int r, double* vs,
                         double* Fs, double* sum_logFs, int fc_steps = 0,
                         double* d_fc = nullptr) {
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
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs, fc_steps, d_fc);
        break;
      case 2:
        batched_kalman_loop_kernel<2>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs, fc_steps, d_fc);
        break;
      case 3:
        batched_kalman_loop_kernel<3>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs, fc_steps, d_fc);
        break;
      case 4:
        batched_kalman_loop_kernel<4>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs, fc_steps, d_fc);
        break;
      case 5:
        batched_kalman_loop_kernel<5>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs, fc_steps, d_fc);
        break;
      case 6:
        batched_kalman_loop_kernel<6>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs, fc_steps, d_fc);
        break;
      case 7:
        batched_kalman_loop_kernel<7>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs, fc_steps, d_fc);
        break;
      case 8:
        batched_kalman_loop_kernel<8>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RRT.raw_data(), P0.raw_data(),
            alpha.raw_data(), batch_size, vs, Fs, sum_logFs, fc_steps, d_fc);
        break;
    }
    CUDA_CHECK(cudaPeekAtLastError());
  } else {
    // Note: not always used
    MLCommon::Sparse::Batched::CSR<double> T_sparse =
      MLCommon::Sparse::Batched::CSR<double>::from_dense(
        T, T_mask, handle.getImpl().getcusolverSpHandle());
    _batched_kalman_loop_large(ys, nobs, T, Z, RRT, P0, alpha, T_sparse, r, vs,
                               Fs, sum_logFs, fc_steps, d_fc);
  }
}

template <int NUM_THREADS>
__global__ void batched_kalman_loglike_kernel(const double* d_vs,
                                              const double* d_Fs,
                                              const double* d_sumLogFs,
                                              int nobs, int batch_size,
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
    if (it + tid < nobs) {
      double _vi = d_vs[idx];
      d_vs2_Fs = _vi * _vi / d_Fs[idx];
    }
    __syncthreads();
    double partial_sum = BlockReduce(temp_storage).Sum(d_vs2_Fs, nobs - it);
    bid_sigma2 += partial_sum;
  }
  if (tid == 0) {
    bid_sigma2 /= nobs;
    loglike[bid] = -.5 * (d_sumLogFs[bid] + nobs * log(bid_sigma2)) -
                   nobs / 2. * (log(2 * M_PI) + 1);
  }
}

void batched_kalman_loglike(const double* d_vs, const double* d_Fs,
                            const double* d_sumLogFs, int nobs, int batch_size,
                            double* loglike, cudaStream_t stream) {
  constexpr int NUM_THREADS = 128;
  batched_kalman_loglike_kernel<NUM_THREADS>
    <<<batch_size, NUM_THREADS, 0, stream>>>(d_vs, d_Fs, d_sumLogFs, nobs,
                                             batch_size, loglike);
  CUDA_CHECK(cudaGetLastError());
}

// Internal Kalman filter implementation that assumes data exists on GPU.
void _batched_kalman_filter(cumlHandle& handle, const double* d_ys, int nobs,
                            const MLCommon::LinAlg::Batched::Matrix<double>& Zb,
                            const MLCommon::LinAlg::Batched::Matrix<double>& Tb,
                            const MLCommon::LinAlg::Batched::Matrix<double>& Rb,
                            std::vector<bool>& T_mask, int r, double* d_vs,
                            double* d_Fs, double* d_loglike,
                            const double* d_sigma2, int fc_steps = 0,
                            double* d_fc = nullptr) {
  const size_t batch_size = Zb.batches();
  auto stream = handle.getStream();
  auto cublasHandle = handle.getImpl().getCublasHandle();
  auto allocator = handle.getDeviceAllocator();

  auto counting = thrust::make_counting_iterator(0);

  MLCommon::LinAlg::Batched::Matrix<double> RQb(r, 1, batch_size, cublasHandle,
                                                allocator, stream, true);
  double* d_RQ = RQb.raw_data();
  const double* d_R = Rb.raw_data();
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     double sigma2 = d_sigma2[bid];
                     for (int i = 0; i < r; i++) {
                       d_RQ[bid * r + i] = d_R[bid * r + i] * sigma2;
                     }
                   });
  MLCommon::LinAlg::Batched::Matrix<double> RRT =
    MLCommon::LinAlg::Batched::b_gemm(RQb, Rb, false, true);

  // Durbin Koopman "Time Series Analysis" pg 138
  ML::PUSH_RANGE("Init P");
  MLCommon::LinAlg::Batched::Matrix<double> P =
    MLCommon::LinAlg::Batched::b_lyapunov(Tb, RRT);
  ML::POP_RANGE();

  // init alpha to zero
  MLCommon::LinAlg::Batched::Matrix<double> alpha(
    r, 1, batch_size, handle.getImpl().getCublasHandle(),
    handle.getDeviceAllocator(), stream, true);

  // init vs, Fs
  // In batch-major format.
  double* d_sumlogFs;

  d_sumlogFs = (double*)handle.getDeviceAllocator()->allocate(
    sizeof(double) * batch_size, stream);

  batched_kalman_loop(handle, d_ys, nobs, Tb, Zb, RRT, P, alpha, T_mask, r,
                      d_vs, d_Fs, d_sumlogFs, fc_steps, d_fc);

  // Finalize loglikelihood
  batched_kalman_loglike(d_vs, d_Fs, d_sumlogFs, nobs, batch_size, d_loglike,
                         stream);

  handle.getDeviceAllocator()->deallocate(d_sumlogFs,
                                          sizeof(double) * batch_size, stream);
}

void init_batched_kalman_matrices(cumlHandle& handle, const double* d_ar,
                                  const double* d_ma, const double* d_sar,
                                  const double* d_sma, int nb,
                                  const ARIMAOrder& order, int r, double* d_Z_b,
                                  double* d_R_b, double* d_T_b,
                                  std::vector<bool>& T_mask) {
  ML::PUSH_RANGE(__func__);

  auto stream = handle.getStream();

  // Note: Z is unused yet but kept to avoid reintroducing it later when
  // adding support for exogeneous variables
  cudaMemsetAsync(d_Z_b, 0.0, r * nb * sizeof(double), stream);
  cudaMemsetAsync(d_R_b, 0.0, r * nb * sizeof(double), stream);
  cudaMemsetAsync(d_T_b, 0.0, r * r * nb * sizeof(double), stream);

  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                   [=] __device__(int bid) {
                     // See TSA pg. 54 for Z,R,T matrices
                     // Z = [1 0 0 0 ... 0]
                     d_Z_b[bid * r] = 1.0;

                     //     |1.0        |
                     // R = |theta_1    |
                     //     | ...       |
                     //     |theta_{r-1}|
                     //
                     d_R_b[bid * r] = 1.0;
                     for (int i = 0; i < r - 1; i++) {
                       d_R_b[bid * r + i + 1] =
                         MLCommon::TimeSeries::reduced_polynomial<false>(
                           bid, d_ma, order.q, d_sma, order.Q, order.s, i + 1);
                     }

                     //     |phi_1  1.0  0.0  ...  0.0|
                     //     | .          1.0          |
                     //     | .              .        |
                     // T = | .                .   0.0|
                     //     | .                  .    |
                     //     | .                    1.0|
                     //     |phi_r  0.0  0.0  ...  0.0|
                     //
                     double* batch_T = d_T_b + bid * r * r;
                     for (int i = 0; i < r; i++) {
                       batch_T[i] =
                         MLCommon::TimeSeries::reduced_polynomial<true>(
                           bid, d_ar, order.p, d_sar, order.P, order.s, i + 1);
                     }
                     // shifted identity
                     for (int i = 0; i < r - 1; i++) {
                       batch_T[(i + 1) * r + i] = 1.0;
                     }

                     // If r=2 and phi_2=-1, I-TxT is singular
                     if (r == 2 && order.p == 2 && abs(batch_T[1] + 1) < 0.01) {
                       batch_T[1] = -0.99;
                     }
                   });

  T_mask.resize(r * r, false);
  for (int iP = 0; iP < order.P + 1; iP++) {
    for (int ip = 0; ip < order.p + 1; ip++) {
      int i = iP * order.s + ip - 1;
      if (i >= 0) T_mask[i] = true;
    }
  }
  for (int i = 0; i < r - 1; i++) {
    T_mask[(i + 1) * r + i] = true;
  }

  ML::POP_RANGE();
}

void batched_kalman_filter(cumlHandle& handle, const double* d_ys, int nobs,
                           const ARIMAParams<double>& params,
                           const ARIMAOrder& order, int batch_size,
                           double* loglike, double* d_vs, bool host_loglike,
                           int fc_steps, double* d_fc) {
  ML::PUSH_RANGE(__func__);

  const size_t ys_len = nobs;

  auto cublasHandle = handle.getImpl().getCublasHandle();
  auto stream = handle.getStream();
  auto allocator = handle.getDeviceAllocator();

  // see (3.18) in TSA by D&K
  int r = order.r();

  MLCommon::LinAlg::Batched::Matrix<double> Zb(1, r, batch_size, cublasHandle,
                                               allocator, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> Tb(r, r, batch_size, cublasHandle,
                                               allocator, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> Rb(r, 1, batch_size, cublasHandle,
                                               allocator, stream, false);

  std::vector<bool> T_mask;
  init_batched_kalman_matrices(handle, params.ar, params.ma, params.sar,
                               params.sma, batch_size, order, r, Zb.raw_data(),
                               Rb.raw_data(), Tb.raw_data(), T_mask);

  ////////////////////////////////////////////////////////////
  // Computation

  double* d_Fs =
    (double*)allocator->allocate(ys_len * batch_size * sizeof(double), stream);

  /* Create log-likelihood device array if host pointer is provided */
  double* d_loglike;
  if (host_loglike) {
    d_loglike =
      (double*)allocator->allocate(batch_size * sizeof(double), stream);
  } else {
    d_loglike = loglike;
  }

  _batched_kalman_filter(handle, d_ys, nobs, Zb, Tb, Rb, T_mask, r, d_vs, d_Fs,
                         d_loglike, params.sigma2, fc_steps, d_fc);

  if (host_loglike) {
    /* Tranfer log-likelihood device -> host */
    MLCommon::updateHost(loglike, d_loglike, batch_size, stream);
    allocator->deallocate(d_loglike, batch_size * sizeof(double), stream);
  }

  allocator->deallocate(d_Fs, ys_len * batch_size * sizeof(double), stream);

  ML::POP_RANGE();
}

void batched_jones_transform(cumlHandle& handle, const ARIMAOrder& order,
                             int batch_size, bool isInv, const double* h_params,
                             double* h_Tparams) {
  int N = order.complexity();
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_params =
    (double*)allocator->allocate(N * batch_size * sizeof(double), stream);
  double* d_Tparams =
    (double*)allocator->allocate(N * batch_size * sizeof(double), stream);
  ARIMAParams<double> params, Tparams;
  params.allocate(order, batch_size, allocator, stream, false);
  Tparams.allocate(order, batch_size, allocator, stream, true);

  MLCommon::updateDevice(d_params, h_params, N * batch_size, stream);

  params.unpack(order, batch_size, d_params, stream);

  MLCommon::TimeSeries::batched_jones_transform(
    order, batch_size, isInv, params, Tparams, allocator, stream);
  Tparams.mu = params.mu;

  Tparams.pack(order, batch_size, d_Tparams, stream);

  MLCommon::updateHost(h_Tparams, d_Tparams, N * batch_size, stream);

  allocator->deallocate(d_params, N * batch_size * sizeof(double), stream);
  allocator->deallocate(d_Tparams, N * batch_size * sizeof(double), stream);
  params.deallocate(order, batch_size, allocator, stream, false);
  Tparams.deallocate(order, batch_size, allocator, stream, true);
}

}  // namespace ML
