/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cuml/tsa/batched_kalman.hpp>

#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <common/nvtx.hpp>
#include <linalg/batched/matrix.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/binary_op.cuh>
#include <sparse/batched/csr.cuh>
#include <timeSeries/arima_helpers.cuh>

namespace ML {

//! Thread-local Matrix-Vector multiplication.
template <int n>
DI void Mv_l(const double* A, const double* v, double* out) {
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = sum;
  }
}

template <int n>
DI void Mv_l(double alpha, const double* A, const double* v, double* out) {
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = alpha * sum;
  }
}

//! Thread-local Matrix-Matrix multiplication.
template <int n, bool aT = false, bool bT = false>
DI void MM_l(const double* A, const double* B, double* out) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        double Aik = aT ? A[k + i * n] : A[i + k * n];
        double Bkj = bT ? B[j + k * n] : B[k + j * n];
        sum += Aik * Bkj;
      }
      out[i + j * n] = sum;
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
 * @param[in]  RQR        Batched R*Q*R'                        (r x r)
 * @param[in]  P          Batched P                             (r x r)
 * @param[in]  alpha      Batched state vector                  (r x 1)
 * @param[in]  intercept  Do we fit an intercept?
 * @param[in]  d_mu       Batched intercept                     (1)
 * @param[in]  batch_size Batch size
 * @param[out] vs         Batched residuals                     (nobs)
 * @param[out] Fs         Batched variance of prediction errors (nobs)    
 * @param[out] sum_logFs  Batched sum of the logs of Fs         (1)
 * @param[in]  n_diff       d + s*D
 * @param[in]  fc_steps   Number of steps to forecast
 * @param[out] d_fc       Array to store the forecast
 * @param[in]  conf_int   Whether to compute confidence intervals
 * @param[in]  d_F_fc     Batched variance of forecast errors   (fc_steps)
 */
template <int rd>
__global__ void batched_kalman_loop_kernel(
  const double* ys, int nobs, const double* T, const double* Z,
  const double* RQR, const double* P, const double* alpha, bool intercept,
  const double* d_mu, int batch_size, double* vs, double* Fs, double* sum_logFs,
  int n_diff, int fc_steps = 0, double* d_fc = nullptr, bool conf_int = false,
  double* d_F_fc = nullptr) {
  constexpr int rd2 = rd * rd;
  double l_RQR[rd2];
  double l_T[rd2];
  double l_Z[rd];
  double l_P[rd2];
  double l_alpha[rd];
  double l_K[rd];
  double l_tmp[rd2];
  double l_TP[rd2];

  int bid = blockDim.x * blockIdx.x + threadIdx.x;

  if (bid < batch_size) {
    // Load global mem into registers
    {
      int b_rd_offset = bid * rd;
      int b_rd2_offset = bid * rd2;
      for (int i = 0; i < rd2; i++) {
        l_RQR[i] = RQR[b_rd2_offset + i];
        l_T[i] = T[b_rd2_offset + i];
        l_P[i] = P[b_rd2_offset + i];
      }
      for (int i = 0; i < rd; i++) {
        if (n_diff > 0) l_Z[i] = Z[b_rd_offset + i];
        l_alpha[i] = alpha[b_rd_offset + i];
      }
    }

    double b_sum_logFs = 0.0;
    const double* b_ys = ys + bid * nobs;
    double* b_vs = vs + bid * nobs;
    double* b_Fs = Fs + bid * nobs;

    double mu = intercept ? d_mu[bid] : 0.0;

    for (int it = 0; it < nobs; it++) {
      // 1. v = y - Z*alpha
      double vs_it = b_ys[it];
      if (n_diff == 0)
        vs_it -= l_alpha[0];
      else {
        for (int i = 0; i < rd; i++) {
          vs_it -= l_alpha[i] * l_Z[i];
        }
      }
      b_vs[it] = vs_it;

      // 2. F = Z*P*Z'
      double _Fs;
      if (n_diff == 0)
        _Fs = l_P[0];
      else {
        _Fs = 0.0;
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
          }
        }
      }
      b_Fs[it] = _Fs;
      if (it >= n_diff) b_sum_logFs += log(_Fs);

      // 3. K = 1/Fs[it] * T*P*Z'
      // TP = T*P
      MM_l<rd>(l_T, l_P, l_TP);
      // K = 1/Fs[it] * TP*Z'
      double _1_Fs = 1.0 / _Fs;
      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_K[i] = _1_Fs * l_TP[i];
        }
      } else
        Mv_l<rd>(_1_Fs, l_TP, l_Z, l_K);

      // 4. alpha = T*alpha + K*vs[it] + c
      // tmp = T*alpha
      Mv_l<rd>(l_T, l_alpha, l_tmp);
      // alpha = tmp + K*vs[it]
      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i] + l_K[i] * vs_it;
      }
      // alpha = alpha + c
      l_alpha[n_diff] += mu;

      // 5. L = T - K * Z
      // L = T (L is tmp)
      for (int i = 0; i < rd2; i++) {
        l_tmp[i] = l_T[i];
      }
      // L = L - K * Z
      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_tmp[i] -= l_K[i];
        }
      } else {
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            l_tmp[j * rd + i] -= l_K[i] * l_Z[j];
          }
        }
      }

      // 6. P = T*P*L' + R*Q*R'
      // P = TP*L'
      MM_l<rd, false, true>(l_TP, l_tmp, l_P);
      // P = P + RQR
      for (int i = 0; i < rd2; i++) {
        l_P[i] += l_RQR[i];
      }
    }
    sum_logFs[bid] = b_sum_logFs;

    // Forecast
    {
      double* b_fc = fc_steps ? d_fc + bid * fc_steps : nullptr;
      double* b_F_fc = conf_int ? d_F_fc + bid * fc_steps : nullptr;
      for (int it = 0; it < fc_steps; it++) {
        if (n_diff == 0)
          b_fc[it] = l_alpha[0];
        else {
          double pred = 0.0;
          for (int i = 0; i < rd; i++) {
            pred += l_alpha[i] * l_Z[i];
          }
          b_fc[it] = pred;
        }

        // alpha = T*alpha + c
        Mv_l<rd>(l_T, l_alpha, l_tmp);
        for (int i = 0; i < rd; i++) {
          l_alpha[i] = l_tmp[i];
        }
        l_alpha[n_diff] += mu;

        if (conf_int) {
          if (n_diff == 0)
            b_F_fc[it] = l_P[0];
          else {
            double _Fs = 0.0;
            for (int i = 0; i < rd; i++) {
              for (int j = 0; j < rd; j++) {
                _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
              }
            }
            b_F_fc[it] = _Fs;
          }

          // P = T*P*T' + RR'
          // TP = T*P
          MM_l<rd>(l_T, l_P, l_TP);
          // P = TP*T'
          MM_l<rd, false, true>(l_TP, l_T, l_P);
          // P = P + RR'
          for (int i = 0; i < rd2; i++) {
            l_P[i] += l_RQR[i];
          }
        }
      }
    }
  }
}

/**
 * Kalman loop for large matrices (r > 8).
 *
 * @param[in]  arima_mem    Pre-allocated temporary memory
 * @param[in]  d_ys         Batched time series
 * @param[in]  nobs         Number of observation per series
 * @param[in]  T            Batched transition matrix.            (r x r)
 * @param[in]  T_sparse     Batched sparse matrix T               (r x r)
 * @param[in]  Z            Batched "design" vector               (1 x r)
 * @param[in]  RQR          Batched R*Q*R'                        (r x r)
 * @param[in]  P            Batched P                             (r x r)
 * @param[in]  alpha        Batched state vector                  (r x 1)
 * @param[in]  intercept    Do we fit an intercept?
 * @param[in]  d_mu         Batched intercept                     (1)
 * @param[in]  r            Dimension of the state vector
 * @param[out] d_vs         Batched residuals                     (nobs)
 * @param[out] d_Fs         Batched variance of prediction errors (nobs)    
 * @param[out] d_sum_logFs  Batched sum of the logs of Fs         (1)
 * @param[in]  n_diff         d + s*D
 * @param[in]  fc_steps     Number of steps to forecast
 * @param[out] d_fc         Array to store the forecast
 * @param[in]  conf_int     Whether to compute confidence intervals
 * @param[out] d_F_fc       Batched variance of forecast errors   (fc_steps)
 */
void _batched_kalman_loop_large(
  const ARIMAMemory<double>& arima_mem, const double* d_ys, int nobs,
  const MLCommon::LinAlg::Batched::Matrix<double>& T,
  const MLCommon::Sparse::Batched::CSR<double>& T_sparse,
  const MLCommon::LinAlg::Batched::Matrix<double>& Z,
  const MLCommon::LinAlg::Batched::Matrix<double>& RQR,
  MLCommon::LinAlg::Batched::Matrix<double>& P,
  MLCommon::LinAlg::Batched::Matrix<double>& alpha, bool intercept,
  const double* d_mu, int rd, double* d_vs, double* d_Fs, double* d_sum_logFs,
  int n_diff, int fc_steps = 0, double* d_fc = nullptr, bool conf_int = false,
  double* d_F_fc = nullptr) {
  auto stream = T.stream();
  auto allocator = T.allocator();
  auto cublasHandle = T.cublasHandle();
  int nb = T.batches();
  int rd2 = rd * rd;
  auto counting = thrust::make_counting_iterator(0);

  // Temporary matrices and vectors
  MLCommon::LinAlg::Batched::Matrix<double> v_tmp(
    rd, 1, nb, cublasHandle, arima_mem.v_tmp_batches, arima_mem.v_tmp_dense,
    allocator, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> m_tmp(
    rd, rd, nb, cublasHandle, arima_mem.m_tmp_batches, arima_mem.m_tmp_dense,
    allocator, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> K(
    rd, 1, nb, cublasHandle, arima_mem.K_batches, arima_mem.K_dense, allocator,
    stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> TP(
    rd, rd, nb, cublasHandle, arima_mem.TP_batches, arima_mem.TP_dense,
    allocator, stream, false);

  // Shortcuts
  const double* d_Z = Z.raw_data();
  double* d_P = P.raw_data();
  double* d_alpha = alpha.raw_data();
  double* d_K = K.raw_data();
  double* d_TP = TP.raw_data();
  double* d_m_tmp = m_tmp.raw_data();
  double* d_v_tmp = v_tmp.raw_data();

  CUDA_CHECK(cudaMemsetAsync(d_sum_logFs, 0, sizeof(double) * nb, stream));

  for (int it = 0; it < nobs; it++) {
    // 1. & 2.
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       const double* b_P = d_P + bid * rd2;
                       const double* b_Z = d_Z + bid * rd;
                       const double* b_alpha = d_alpha + bid * rd;

                       double vt = d_ys[bid * nobs + it];
                       if (n_diff == 0) {
                         vt -= b_alpha[0];
                       } else {
                         for (int i = 0; i < rd; i++) {
                           vt -= b_alpha[i] * b_Z[i];
                         }
                       }
                       d_vs[bid * nobs + it] = vt;

                       double _F;
                       if (n_diff == 0)
                         _F = b_P[0];
                       else {
                         _F = 0.0;
                         for (int i = 0; i < rd; i++) {
                           for (int j = 0; j < rd; j++) {
                             _F += b_P[j * rd + i] * b_Z[i] * b_Z[j];
                           }
                         }
                       }
                       d_Fs[bid * nobs + it] = _F;
                       if (it >= n_diff) d_sum_logFs[bid] += log(_F);
                     });

    // 3. K = 1/Fs[it] * T*P*Z'
    // TP = T*P (also used later)
    if (rd <= 32)
      MLCommon::Sparse::Batched::b_spmm(1.0, T_sparse, P, 0.0, TP);
    else
      MLCommon::LinAlg::Batched::b_gemm(false, false, rd, rd, rd, 1.0, T, P,
                                        0.0, TP);
    // K = 1/Fs[it] * TP*Z'
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       const double* b_TP = d_TP + bid * rd2;
                       double* b_K = d_K + bid * rd;

                       double _1_Fs = 1.0 / d_Fs[bid * nobs + it];
                       if (n_diff == 0) {
                         for (int i = 0; i < rd; i++) {
                           b_K[i] = _1_Fs * b_TP[i];
                         }
                       } else {
                         const double* b_Z = d_Z + bid * rd;
                         for (int i = 0; i < rd; i++) {
                           double acc = 0.0;
                           for (int j = 0; j < rd; j++) {
                             acc += b_TP[rd * j + i] * b_Z[j];
                           }
                           b_K[i] = _1_Fs * acc;
                         }
                       }
                     });

    // 4. alpha = T*alpha + K*vs[it] + c
    // v_tmp = T*alpha
    MLCommon::Sparse::Batched::b_spmv(1.0, T_sparse, alpha, 0.0, v_tmp);
    // alpha = v_tmp + K*vs[it] + c
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       const double* b_Talpha = d_v_tmp + bid * rd;
                       const double* b_K = d_K + bid * rd;
                       double* b_alpha = d_alpha + bid * rd;

                       double _vs = d_vs[bid * nobs + it];
                       for (int i = 0; i < rd; i++) {
                         double mu =
                           (intercept && i == n_diff) ? d_mu[bid] : 0.0;
                         b_alpha[i] = b_Talpha[i] + b_K[i] * _vs + mu;
                       }
                     });

    // 5. L = T - K * Z
    // L = T (L is m_tmp)
    raft::copy(m_tmp.raw_data(), T.raw_data(), nb * rd2, stream);
    // L = L - K * Z
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       const double* b_K = d_K + bid * rd;
                       double* b_L = d_m_tmp + bid * rd2;

                       if (n_diff == 0) {
                         for (int i = 0; i < rd; i++) {
                           b_L[i] -= b_K[i];
                         }
                       } else {
                         const double* b_Z = d_Z + bid * rd;
                         for (int i = 0; i < rd; i++) {
                           for (int j = 0; j < rd; j++) {
                             b_L[j * rd + i] -= b_K[i] * b_Z[j];
                           }
                         }
                       }
                     });
    // MLCommon::LinAlg::Batched::b_gemm(false, false, rd, rd, 1, -1.0, K, Z, 1.0,
    //                                   m_tmp);  // generic

    // 6. P = T*P*L' + R*Q*R'
    // P = TP*L'
    MLCommon::LinAlg::Batched::b_gemm(false, true, rd, rd, rd, 1.0, TP, m_tmp,
                                      0.0, P);
    // P = P + R*Q*R'
    raft::linalg::binaryOp(
      d_P, d_P, RQR.raw_data(), rd2 * nb,
      [=] __device__(double a, double b) { return a + b; }, stream);
  }

  // Forecast
  for (int it = 0; it < fc_steps; it++) {
    thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                     [=] __device__(int bid) {
                       const double* b_alpha = d_alpha + bid * rd;

                       double pred;
                       if (n_diff == 0) {
                         pred = b_alpha[0];
                       } else {
                         const double* b_Z = d_Z + bid * rd;

                         pred = 0.0;
                         for (int i = 0; i < rd; i++) {
                           pred += b_alpha[i] * b_Z[i];
                         }
                       }
                       d_fc[bid * fc_steps + it] = pred;
                     });

    // alpha = T*alpha + c
    // alpha = T*alpha
    MLCommon::Sparse::Batched::b_spmv(1.0, T_sparse, alpha, 0.0, v_tmp);
    raft::copy(d_alpha, v_tmp.raw_data(), rd * nb, stream);
    // alpha += c
    if (intercept) {
      thrust::for_each(
        thrust::cuda::par.on(stream), counting, counting + nb,
        [=] __device__(int bid) { d_alpha[bid * rd + n_diff] += d_mu[bid]; });
    }

    if (conf_int) {
      thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                       [=] __device__(int bid) {
                         const double* b_P = d_P + bid * rd2;

                         double Ft;
                         if (n_diff == 0)
                           Ft = b_P[0];
                         else {
                           const double* b_Z = d_Z + bid * rd;
                           Ft = 0.0;
                           for (int i = 0; i < rd; i++) {
                             for (int j = 0; j < rd; j++) {
                               Ft += b_P[j * rd + i] * b_Z[i] * b_Z[j];
                             }
                           }
                         }

                         d_F_fc[bid * fc_steps + it] = Ft;
                       });

      // P = T*P*T' + R*Q*R'
      // TP = T*P
      if (rd <= 32)
        MLCommon::Sparse::Batched::b_spmm(1.0, T_sparse, P, 0.0, TP);
      else
        MLCommon::LinAlg::Batched::b_gemm(false, false, rd, rd, rd, 1.0, T, P,
                                          0.0, TP);
      // P = TP*T'
      MLCommon::LinAlg::Batched::b_gemm(false, true, rd, rd, rd, 1.0, TP, T,
                                        0.0, P);
      // P = P + R*Q*R'
      raft::linalg::binaryOp(
        d_P, d_P, RQR.raw_data(), rd2 * nb,
        [=] __device__(double a, double b) { return a + b; }, stream);
    }
  }
}

/// Wrapper around functions that execute the Kalman loop (for performance)
void batched_kalman_loop(raft::handle_t& handle,
                         const ARIMAMemory<double>& arima_mem, const double* ys,
                         int nobs,
                         const MLCommon::LinAlg::Batched::Matrix<double>& T,
                         const MLCommon::LinAlg::Batched::Matrix<double>& Z,
                         const MLCommon::LinAlg::Batched::Matrix<double>& RQR,
                         MLCommon::LinAlg::Batched::Matrix<double>& P0,
                         MLCommon::LinAlg::Batched::Matrix<double>& alpha,
                         std::vector<bool>& T_mask, bool intercept,
                         const double* d_mu, const ARIMAOrder& order,
                         double* vs, double* Fs, double* sum_logFs,
                         int fc_steps = 0, double* d_fc = nullptr,
                         bool conf_int = false, double* d_F_fc = nullptr) {
  const int batch_size = T.batches();
  auto stream = T.stream();
  int rd = order.rd();
  int n_diff = order.n_diff();
  dim3 numThreadsPerBlock(32, 1);
  dim3 numBlocks(raft::ceildiv<int>(batch_size, numThreadsPerBlock.x), 1);
  if (rd <= 8) {
    switch (rd) {
      case 1:
        batched_kalman_loop_kernel<1>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RQR.raw_data(), P0.raw_data(),
            alpha.raw_data(), intercept, d_mu, batch_size, vs, Fs, sum_logFs,
            n_diff, fc_steps, d_fc, conf_int, d_F_fc);
        break;
      case 2:
        batched_kalman_loop_kernel<2>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RQR.raw_data(), P0.raw_data(),
            alpha.raw_data(), intercept, d_mu, batch_size, vs, Fs, sum_logFs,
            n_diff, fc_steps, d_fc, conf_int, d_F_fc);
        break;
      case 3:
        batched_kalman_loop_kernel<3>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RQR.raw_data(), P0.raw_data(),
            alpha.raw_data(), intercept, d_mu, batch_size, vs, Fs, sum_logFs,
            n_diff, fc_steps, d_fc, conf_int, d_F_fc);
        break;
      case 4:
        batched_kalman_loop_kernel<4>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RQR.raw_data(), P0.raw_data(),
            alpha.raw_data(), intercept, d_mu, batch_size, vs, Fs, sum_logFs,
            n_diff, fc_steps, d_fc, conf_int, d_F_fc);
        break;
      case 5:
        batched_kalman_loop_kernel<5>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RQR.raw_data(), P0.raw_data(),
            alpha.raw_data(), intercept, d_mu, batch_size, vs, Fs, sum_logFs,
            n_diff, fc_steps, d_fc, conf_int, d_F_fc);
        break;
      case 6:
        batched_kalman_loop_kernel<6>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RQR.raw_data(), P0.raw_data(),
            alpha.raw_data(), intercept, d_mu, batch_size, vs, Fs, sum_logFs,
            n_diff, fc_steps, d_fc, conf_int, d_F_fc);
        break;
      case 7:
        batched_kalman_loop_kernel<7>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RQR.raw_data(), P0.raw_data(),
            alpha.raw_data(), intercept, d_mu, batch_size, vs, Fs, sum_logFs,
            n_diff, fc_steps, d_fc, conf_int, d_F_fc);
        break;
      case 8:
        batched_kalman_loop_kernel<8>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            ys, nobs, T.raw_data(), Z.raw_data(), RQR.raw_data(), P0.raw_data(),
            alpha.raw_data(), intercept, d_mu, batch_size, vs, Fs, sum_logFs,
            n_diff, fc_steps, d_fc, conf_int, d_F_fc);
        break;
    }
    CUDA_CHECK(cudaPeekAtLastError());
  } else {
    // Note: not always used
    MLCommon::Sparse::Batched::CSR<double> T_sparse =
      MLCommon::Sparse::Batched::CSR<double>::from_dense(
        T, T_mask, handle.get_cusolver_sp_handle(), arima_mem.T_values,
        arima_mem.T_col_index, arima_mem.T_row_index);
    _batched_kalman_loop_large(arima_mem, ys, nobs, T, T_sparse, Z, RQR, P0,
                               alpha, intercept, d_mu, rd, vs, Fs, sum_logFs,
                               n_diff, fc_steps, d_fc, conf_int, d_F_fc);
  }
}

template <int NUM_THREADS>
__global__ void batched_kalman_loglike_kernel(
  const double* d_vs, const double* d_Fs, const double* d_sumLogFs, int nobs,
  int batch_size, double* d_loglike, double* d_sigma2, int n_diff,
  double level) {
  using BlockReduce = cub::BlockReduce<double, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  double bid_sigma2 = 0.0;
  for (int it = 0; it < nobs; it += NUM_THREADS) {
    // vs and Fs are in time-major order (memory layout: column major)
    int idx = (it + tid) + bid * nobs;
    double d_vs2_Fs = 0.0;
    if (it + tid >= n_diff && it + tid < nobs) {
      double _vi = d_vs[idx];
      d_vs2_Fs = _vi * _vi / d_Fs[idx];
    }
    __syncthreads();
    double partial_sum = BlockReduce(temp_storage).Sum(d_vs2_Fs, nobs - it);
    bid_sigma2 += partial_sum;
  }
  if (tid == 0) {
    double nobs_diff_f = static_cast<double>(nobs - n_diff);
    bid_sigma2 /= nobs_diff_f;
    if (level != 0) d_sigma2[bid] = bid_sigma2;
    d_loglike[bid] = -.5 * (d_sumLogFs[bid] + nobs_diff_f * bid_sigma2 +
                            nobs_diff_f * (log(2 * M_PI)));
  }
}

/**
 * Kernel to finalize the computation of confidence intervals
 *
 * @note: One block per batch member, one thread per forecast time step
 *
 * @param[in]    d_fc       Mean forecasts
 * @param[in]    d_sigma2   sum(v_t * v_t / F_t) / n_obs_diff
 * @param[inout] d_lower    Input: F_{n+t}
 *                          Output: lower bound of the confidence intervals
 * @param[out]   d_upper    Upper bound of the confidence intervals
 * @param[in]    fc_steps   Number of forecast steps
 * @param[in]    multiplier Coefficient associated with the confidence level
 */
__global__ void confidence_intervals(const double* d_fc, const double* d_sigma2,
                                     double* d_lower, double* d_upper,
                                     int fc_steps, double multiplier) {
  int idx = blockIdx.x * fc_steps + threadIdx.x;
  double fc = d_fc[idx];
  double margin = multiplier * sqrt(d_lower[idx] * d_sigma2[blockIdx.x]);
  d_lower[idx] = fc - margin;
  d_upper[idx] = fc + margin;
}

void _lyapunov_wrapper(raft::handle_t& handle,
                       const ARIMAMemory<double>& arima_mem,
                       const MLCommon::LinAlg::Batched::Matrix<double>& A,
                       MLCommon::LinAlg::Batched::Matrix<double>& Q,
                       MLCommon::LinAlg::Batched::Matrix<double>& X, int r) {
  if (r <= 5) {
    auto stream = handle.get_stream();
    auto cublasHandle = handle.get_cublas_handle();
    auto allocator = handle.get_device_allocator();
    int batch_size = A.batches();
    int r2 = r * r;

    //
    // Use direct solution with Kronecker product
    //

    MLCommon::LinAlg::Batched::Matrix<double> I_m_AxA(
      r2, r2, batch_size, cublasHandle, arima_mem.I_m_AxA_batches,
      arima_mem.I_m_AxA_dense, allocator, stream, false);
    MLCommon::LinAlg::Batched::Matrix<double> I_m_AxA_inv(
      r2, r2, batch_size, cublasHandle, arima_mem.I_m_AxA_inv_batches,
      arima_mem.I_m_AxA_inv_dense, allocator, stream, false);

    MLCommon::LinAlg::Batched::_direct_lyapunov_helper(
      A, Q, X, I_m_AxA, I_m_AxA_inv, arima_mem.I_m_AxA_P,
      arima_mem.I_m_AxA_info, r);
  } else {
    // Note: the other Lyapunov solver is doing temporary mem allocations,
    // but when r > 5, allocation overhead shouldn't be a bottleneck
    X = MLCommon::LinAlg::Batched::b_lyapunov(A, Q);
  }
}

/// Internal Kalman filter implementation that assumes data exists on GPU.
void _batched_kalman_filter(
  raft::handle_t& handle, const ARIMAMemory<double>& arima_mem,
  const double* d_ys, int nobs, const ARIMAOrder& order,
  const MLCommon::LinAlg::Batched::Matrix<double>& Zb,
  const MLCommon::LinAlg::Batched::Matrix<double>& Tb,
  const MLCommon::LinAlg::Batched::Matrix<double>& Rb,
  std::vector<bool>& T_mask, double* d_vs, double* d_Fs, double* d_loglike,
  const double* d_sigma2, bool intercept, const double* d_mu, int fc_steps,
  double* d_fc, double level, double* d_lower, double* d_upper) {
  const size_t batch_size = Zb.batches();
  auto stream = handle.get_stream();
  auto cublasHandle = handle.get_cublas_handle();
  auto allocator = handle.get_device_allocator();

  auto counting = thrust::make_counting_iterator(0);

  int n_diff = order.n_diff();
  int rd = order.rd();
  int r = order.r();

  MLCommon::LinAlg::Batched::Matrix<double> RQb(
    rd, 1, batch_size, cublasHandle, arima_mem.RQ_batches, arima_mem.RQ_dense,
    allocator, stream, true);
  double* d_RQ = RQb.raw_data();
  const double* d_R = Rb.raw_data();
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     double sigma2 = d_sigma2[bid];
                     for (int i = 0; i < rd; i++) {
                       d_RQ[bid * rd + i] = d_R[bid * rd + i] * sigma2;
                     }
                   });
  MLCommon::LinAlg::Batched::Matrix<double> RQR(
    rd, rd, batch_size, cublasHandle, arima_mem.RQR_batches,
    arima_mem.RQR_dense, allocator, stream, false);
  MLCommon::LinAlg::Batched::b_gemm(false, true, rd, rd, 1, 1.0, RQb, Rb, 0.0,
                                    RQR);

  // Durbin Koopman "Time Series Analysis" pg 138
  ML::PUSH_RANGE("Init P");
  MLCommon::LinAlg::Batched::Matrix<double> P(
    rd, rd, batch_size, cublasHandle, arima_mem.P_batches, arima_mem.P_dense,
    allocator, stream, true);
  {
    double* d_P = P.raw_data();

    if (n_diff > 0) {
      // Initialize the diffuse part with a large variance
      /// TODO: pass this as a parameter
      constexpr double kappa = 1e6;
      thrust::for_each(thrust::cuda::par.on(stream), counting,
                       counting + batch_size, [=] __device__(int bid) {
                         double* b_P = d_P + rd * rd * bid;
                         for (int i = 0; i < n_diff; i++) {
                           b_P[(rd + 1) * i] = kappa;
                         }
                       });

      // Initialize the stationary part by solving a Lyapunov equation
      MLCommon::LinAlg::Batched::Matrix<double> Ts(
        r, r, batch_size, cublasHandle, arima_mem.Ts_batches,
        arima_mem.Ts_dense, allocator, stream, false);
      MLCommon::LinAlg::Batched::Matrix<double> RQRs(
        r, r, batch_size, cublasHandle, arima_mem.RQRs_batches,
        arima_mem.RQRs_dense, allocator, stream, false);
      MLCommon::LinAlg::Batched::Matrix<double> Ps(
        r, r, batch_size, cublasHandle, arima_mem.Ps_batches,
        arima_mem.Ps_dense, allocator, stream, false);

      MLCommon::LinAlg::Batched::b_2dcopy(Tb, Ts, n_diff, n_diff, r, r);
      MLCommon::LinAlg::Batched::b_2dcopy(RQR, RQRs, n_diff, n_diff, r, r);
      // Ps = MLCommon::LinAlg::Batched::b_lyapunov(Ts, RQRs);
      _lyapunov_wrapper(handle, arima_mem, Ts, RQRs, Ps, r);
      MLCommon::LinAlg::Batched::b_2dcopy(Ps, P, 0, 0, r, r, n_diff, n_diff);
    } else {
      // Initialize by solving a Lyapunov equation
      // P = MLCommon::LinAlg::Batched::b_lyapunov(Tb, RQR);
      _lyapunov_wrapper(handle, arima_mem, Tb, RQR, P, rd);
    }
  }
  ML::POP_RANGE();

  // Initialize the state alpha by solving (I - T*) x* = c with:
  //     | mu |
  // c = | 0  |
  //     | .  |
  //     | 0  |
  // T* = T[d+s*D:, d+s*D:]
  // x* = alpha_0[d+s*D:]
  MLCommon::LinAlg::Batched::Matrix<double> alpha(
    rd, 1, batch_size, handle.get_cublas_handle(), arima_mem.alpha_batches,
    arima_mem.alpha_dense, handle.get_device_allocator(), stream, false);
  if (intercept) {
    // Compute I-T*
    MLCommon::LinAlg::Batched::Matrix<double> ImT(
      r, r, batch_size, cublasHandle, arima_mem.ImT_batches,
      arima_mem.ImT_dense, allocator, stream, false);
    const double* d_T = Tb.raw_data();
    double* d_ImT = ImT.raw_data();
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       const double* b_T = d_T + rd * rd * bid;
                       double* b_ImT = d_ImT + r * r * bid;
                       for (int i = 0; i < r; i++) {
                         for (int j = 0; j < r; j++) {
                           b_ImT[r * j + i] =
                             (i == j ? 1.0 : 0.0) -
                             b_T[rd * (j + n_diff) + i + n_diff];
                         }
                       }
                     });

    // For r=1, prevent I-T from being too close to [[0]] -> no solution
    if (r == 1) {
      thrust::for_each(thrust::cuda::par.on(stream), counting,
                       counting + batch_size, [=] __device__(int bid) {
                         if (abs(d_ImT[bid]) < 1e-3)
                           d_ImT[bid] = raft::signPrim(d_ImT[bid]) * 1e-3;
                       });
    }

    // Compute (I-T*)^-1
    MLCommon::LinAlg::Batched::Matrix<double> ImT_inv(
      r, r, batch_size, cublasHandle, arima_mem.ImT_inv_batches,
      arima_mem.ImT_inv_dense, allocator, stream, false);
    MLCommon::LinAlg::Batched::Matrix<double>::inv(
      ImT, ImT_inv, arima_mem.ImT_inv_P, arima_mem.ImT_inv_info);

    // Compute (I-T*)^-1 * c -> multiply 1st column by mu
    const double* d_ImT_inv = ImT_inv.raw_data();
    double* d_alpha = alpha.raw_data();
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       const double* b_ImT_inv = d_ImT_inv + r * r * bid;
                       double* b_alpha = d_alpha + rd * bid;
                       double mu = d_mu[bid];
                       for (int i = 0; i < n_diff; i++) {
                         b_alpha[i] = 0;
                       }
                       for (int i = 0; i < r; i++) {
                         b_alpha[i + n_diff] = b_ImT_inv[i] * mu;
                       }
                     });
  } else {
    // Memset alpha to 0
    CUDA_CHECK(cudaMemsetAsync(alpha.raw_data(), 0,
                               sizeof(double) * rd * batch_size, stream));
  }

  batched_kalman_loop(handle, arima_mem, d_ys, nobs, Tb, Zb, RQR, P, alpha,
                      T_mask, intercept, d_mu, order, d_vs, d_Fs,
                      arima_mem.sumLogF_buffer, fc_steps, d_fc, level > 0,
                      d_lower);

  // Finalize loglikelihood and prediction intervals
  constexpr int NUM_THREADS = 128;
  batched_kalman_loglike_kernel<NUM_THREADS>
    <<<batch_size, NUM_THREADS, 0, stream>>>(
      d_vs, d_Fs, arima_mem.sumLogF_buffer, nobs, batch_size, d_loglike,
      arima_mem.sigma2_buffer, n_diff, level);
  CUDA_CHECK(cudaPeekAtLastError());
  if (level > 0) {
    confidence_intervals<<<batch_size, fc_steps, 0, stream>>>(
      d_fc, arima_mem.sigma2_buffer, d_lower, d_upper, fc_steps,
      sqrt(2.0) * erfinv(level));
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

void init_batched_kalman_matrices(raft::handle_t& handle, const double* d_ar,
                                  const double* d_ma, const double* d_sar,
                                  const double* d_sma, int nb,
                                  const ARIMAOrder& order, int rd,
                                  double* d_Z_b, double* d_R_b, double* d_T_b,
                                  std::vector<bool>& T_mask) {
  ML::PUSH_RANGE(__func__);

  auto stream = handle.get_stream();

  // Note: Z is unused yet but kept to avoid reintroducing it later when
  // adding support for exogeneous variables
  cudaMemsetAsync(d_Z_b, 0.0, rd * nb * sizeof(double), stream);
  cudaMemsetAsync(d_R_b, 0.0, rd * nb * sizeof(double), stream);
  cudaMemsetAsync(d_T_b, 0.0, rd * rd * nb * sizeof(double), stream);

  int n_diff = order.n_diff();
  int r = order.r();

  auto counting = thrust::make_counting_iterator(0);
  auto n_theta = order.n_theta();
  auto n_phi = order.n_phi();
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + nb,
    [=] __device__(int bid) {
      // See TSA pg. 54 for Z, R, T matrices

      // Z = [ 1 | 0 . . 0 1 0 . . 0 1 | 1 0 . . 0 ]
      //       d |         s*D         |     r
      for (int i = 0; i < order.d; i++) d_Z_b[bid * rd + i] = 1.0;
      for (int i = 1; i <= order.D; i++)
        d_Z_b[bid * rd + order.d + i * order.s - 1] = 1.0;
      d_Z_b[bid * rd + n_diff] = 1.0;

      //     |     0     |
      //     |     .     |  d + s*D
      //     |     0     |_ _
      // R = |     1     |
      //     |  theta_1  |  r
      //     |     .     |
      //     |theta_{r-1}|
      //
      d_R_b[bid * rd + n_diff] = 1.0;
      for (int i = 0; i < n_theta; i++) {
        d_R_b[bid * rd + n_diff + i + 1] =
          MLCommon::TimeSeries::reduced_polynomial<false>(
            bid, d_ma, order.q, d_sma, order.Q, order.s, i + 1);
      }

      //     | 1 | 0 .. 0 1 | 1                |  d
      //     |_ _|_ _ _ _ _ |_ _ _ _ _ _ _ _ _ |_ _
      //     |   | 0 .. 0 1 | 1                |
      //     |   | 1      0 |                  |
      //     |   |   .    . |                  |  s*D
      //     |   |    .   . |                  |
      // T = |   | 0    1 0 |                  |
      //     |_ _|_ _ _ _ _ |_ _ _ _ _ _ _ _ _ |_ _
      //     |   |          | phi_1  1         |
      //     |   |          |  .       1    0  |
      //     |   |          |  .         .     |  r
      //     |   |          |  .     0     .   |
      //     |   |          |  .             1 |
      //     |   |          | phi_r  0  . .  0 |
      //
      // (non-comprehensive example with d=1 and D=1)
      //
      double* batch_T = d_T_b + bid * rd * rd;
      // 1. Differencing component
      for (int i = 0; i < order.d; i++) {
        for (int j = i; j < order.d; j++) {
          batch_T[j * rd + i] = 1.0;
        }
      }
      for (int id = 0; id < order.d; id++) {
        batch_T[n_diff * rd + id] = 1.0;
        for (int iD = 1; iD <= order.D; iD++) {
          batch_T[(order.d + order.s * iD - 1) * rd + id] = 1.0;
        }
      }
      // 2. Seasonal differencing component
      for (int iD = 0; iD < order.D; iD++) {
        int offset = order.d + iD * order.s;
        for (int i = 0; i < order.s - 1; i++) {
          batch_T[(offset + i) * rd + offset + i + 1] = 1.0;
        }
        batch_T[(offset + order.s - 1) * rd + offset] = 1.0;
        batch_T[n_diff * rd + offset] = 1.0;
      }
      if (order.D == 2) {
        batch_T[(n_diff - 1) * rd + order.d] = 1.0;
      }
      // 3. Auto-Regressive component
      for (int i = 0; i < n_phi; i++) {
        batch_T[n_diff * (rd + 1) + i] =
          MLCommon::TimeSeries::reduced_polynomial<true>(
            bid, d_ar, order.p, d_sar, order.P, order.s, i + 1);
      }
      for (int i = 0; i < r - 1; i++) {
        batch_T[(n_diff + i + 1) * rd + n_diff + i] = 1.0;
      }

      // If rd=2 and phi_2=-1, I-TxT is singular
      if (rd == 2 && order.p == 2 && abs(batch_T[1] + 1) < 0.01) {
        batch_T[1] = -0.99;
      }
    });

  // T density/sparsity mask
  T_mask.resize(rd * rd, false);
  // 1. Differencing component
  for (int i = 0; i < order.d; i++) {
    for (int j = i; j < order.d; j++) {
      T_mask[j * rd + i] = true;
    }
  }
  for (int id = 0; id < order.d; id++) {
    T_mask[n_diff * rd + id] = true;
    for (int iD = 1; iD <= order.D; iD++) {
      T_mask[(order.d + order.s * iD - 1) * rd + id] = true;
    }
  }
  // 2. Seasonal differencing component
  for (int iD = 0; iD < order.D; iD++) {
    int offset = order.d + iD * order.s;
    for (int i = 0; i < order.s - 1; i++) {
      T_mask[(offset + i) * rd + offset + i + 1] = true;
    }
    T_mask[(offset + order.s - 1) * rd + offset] = true;
    T_mask[n_diff * rd + offset] = true;
  }
  if (order.D == 2) {
    T_mask[(n_diff - 1) * rd + order.d] = true;
  }
  // 3. Auto-Regressive component
  for (int iP = 0; iP < order.P + 1; iP++) {
    for (int ip = 0; ip < order.p + 1; ip++) {
      int i = iP * order.s + ip - 1;
      if (i >= 0) T_mask[n_diff * (rd + 1) + i] = true;
    }
  }
  for (int i = 0; i < r - 1; i++) {
    T_mask[(n_diff + i + 1) * rd + n_diff + i] = true;
  }

  ML::POP_RANGE();
}

void batched_kalman_filter(
  raft::handle_t& handle, const ARIMAMemory<double>& arima_mem,
  const double* d_ys, int nobs, const ARIMAParams<double>& params,
  const ARIMAOrder& order, int batch_size, double* d_loglike, double* d_vs,
  int fc_steps, double* d_fc, double level, double* d_lower, double* d_upper) {
  ML::PUSH_RANGE(__func__);

  auto cublasHandle = handle.get_cublas_handle();
  auto stream = handle.get_stream();
  auto allocator = handle.get_device_allocator();

  // see (3.18) in TSA by D&K
  int rd = order.rd();

  MLCommon::LinAlg::Batched::Matrix<double> Zb(
    1, rd, batch_size, cublasHandle, arima_mem.Z_batches, arima_mem.Z_dense,
    allocator, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> Tb(
    rd, rd, batch_size, cublasHandle, arima_mem.T_batches, arima_mem.T_dense,
    allocator, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> Rb(
    rd, 1, batch_size, cublasHandle, arima_mem.R_batches, arima_mem.R_dense,
    allocator, stream, false);

  std::vector<bool> T_mask;
  init_batched_kalman_matrices(handle, params.ar, params.ma, params.sar,
                               params.sma, batch_size, order, rd, Zb.raw_data(),
                               Rb.raw_data(), Tb.raw_data(), T_mask);

  ////////////////////////////////////////////////////////////
  // Computation

  _batched_kalman_filter(handle, arima_mem, d_ys, nobs, order, Zb, Tb, Rb,
                         T_mask, d_vs, arima_mem.F_buffer, d_loglike,
                         params.sigma2, static_cast<bool>(order.k), params.mu,
                         fc_steps, d_fc, level, d_lower, d_upper);

  ML::POP_RANGE();
}

void batched_jones_transform(raft::handle_t& handle,
                             const ARIMAMemory<double>& arima_mem,
                             const ARIMAOrder& order, int batch_size,
                             bool isInv, const double* h_params,
                             double* h_Tparams) {
  int N = order.complexity();
  auto allocator = handle.get_device_allocator();
  auto stream = handle.get_stream();
  double* d_params = arima_mem.d_params;
  double* d_Tparams = arima_mem.d_Tparams;
  ARIMAParams<double> params = {arima_mem.params_mu,  arima_mem.params_ar,
                                arima_mem.params_ma,  arima_mem.params_sar,
                                arima_mem.params_sma, arima_mem.params_sigma2};
  ARIMAParams<double> Tparams = {
    arima_mem.Tparams_mu,  arima_mem.Tparams_ar,  arima_mem.Tparams_ma,
    arima_mem.Tparams_sar, arima_mem.Tparams_sma, arima_mem.Tparams_sigma2};

  raft::update_device(d_params, h_params, N * batch_size, stream);

  params.unpack(order, batch_size, d_params, stream);

  MLCommon::TimeSeries::batched_jones_transform(
    order, batch_size, isInv, params, Tparams, allocator, stream);
  Tparams.mu = params.mu;

  Tparams.pack(order, batch_size, d_Tparams, stream);

  raft::update_host(h_Tparams, d_Tparams, N * batch_size, stream);
}

}  // namespace ML
