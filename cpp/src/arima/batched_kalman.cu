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

#include <cuml/tsa/batched_kalman.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>

#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/binary_op.cuh>
#include <rmm/device_uvector.hpp>

#include <common/nvtx.hpp>
#include <linalg/batched/matrix.cuh>
#include <linalg/block.cuh>
#include <timeSeries/arima_helpers.cuh>

namespace ML {

//! Thread-local Matrix-Vector multiplication.
template <int n>
DI void Mv_l(const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = sum;
  }
}

template <int n>
DI void Mv_l(double alpha, const double* A, const double* v, double* out)
{
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
DI void MM_l(const double* A, const double* B, double* out)
{
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
 * @param[out] d_F_fc     Batched variance of forecast errors   (fc_steps)
 */
template <int rd>
__global__ void batched_kalman_loop_kernel(const double* ys,
                                           int nobs,
                                           const double* T,
                                           const double* Z,
                                           const double* RQR,
                                           const double* P,
                                           const double* alpha,
                                           bool intercept,
                                           const double* d_mu,
                                           int batch_size,
                                           double* vs,
                                           double* Fs,
                                           double* sum_logFs,
                                           int n_diff,
                                           int fc_steps   = 0,
                                           double* d_fc   = nullptr,
                                           bool conf_int  = false,
                                           double* d_F_fc = nullptr)
{
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
      int b_rd_offset  = bid * rd;
      int b_rd2_offset = bid * rd2;
      for (int i = 0; i < rd2; i++) {
        l_RQR[i] = RQR[b_rd2_offset + i];
        l_T[i]   = T[b_rd2_offset + i];
        l_P[i]   = P[b_rd2_offset + i];
      }
      for (int i = 0; i < rd; i++) {
        if (n_diff > 0) l_Z[i] = Z[b_rd_offset + i];
        l_alpha[i] = alpha[b_rd_offset + i];
      }
    }

    double b_sum_logFs = 0.0;
    const double* b_ys = ys + bid * nobs;
    double* b_vs       = vs + bid * nobs;
    double* b_Fs       = Fs + bid * nobs;

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
      double* b_fc   = fc_steps ? d_fc + bid * fc_steps : nullptr;
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
 * This union allows for efficient reuse of shared memory in the Kalman
 * filter.
 */
template <typename GemmPolicy, typename GemvPolicy, typename T>
union KalmanLoopSharedMemory {
  MLCommon::LinAlg::ReductionStorage<GemmPolicy::BlockSize, T> reduction_storage;
  MLCommon::LinAlg::GemmStorage<GemmPolicy, T> gemm_storage;
  MLCommon::LinAlg::GemvStorage<GemvPolicy, T> gemv_storage[2];
};

/**
 * Kalman loop kernel. Each block computes kalman filter for a single series.
 *
 * @tparam     GemmPolicy  Execution policy for GEMM
 * @tparam     GemvPolicy  Execution policy for GEMV
 * @param[in]  d_ys        Batched time series
 * @param[in]  batch_size  Batch size
 * @param[in]  n_obs       Number of observation per series
 * @param[in]  d_T         Batched transition matrix.            (r x r)
 * @param[in]  d_Z         Batched "design" vector               (1 x r)
 * @param[in]  d_RQR       Batched R*Q*R'                        (r x r)
 * @param[in]  d_P         Batched P                             (r x r)
 * @param[in]  d_alpha     Batched state vector                  (r x 1)
 * @param[in]  d_m_tmp     Batched temporary matrix              (r x r)
 * @param[in]  d_TP        Batched temporary matrix to store TP  (r x r)
 * @param[in]  intercept   Do we fit an intercept?
 * @param[in]  d_mu        Batched intercept                     (1)
 * @param[in]  rd          State vector dimension
 * @param[out] d_vs        Batched residuals                     (nobs)
 * @param[out] d_Fs        Batched variance of prediction errors (nobs)
 * @param[out] d_sum_logFs Batched sum of the logs of Fs         (1)
 * @param[in]  n_diff      d + s*D
 * @param[in]  fc_steps    Number of steps to forecast
 * @param[out] d_fc        Array to store the forecast
 * @param[in]  conf_int    Whether to compute confidence intervals
 * @param[out] d_F_fc      Batched variance of forecast errors   (fc_steps)
 */
template <typename GemmPolicy, typename GemvPolicy>
__global__ void _batched_kalman_device_loop_large_kernel(const double* d_ys,
                                                         int batch_size,
                                                         int n_obs,
                                                         const double* d_T,
                                                         const double* d_Z,
                                                         const double* d_RQR,
                                                         double* d_P,
                                                         double* d_alpha,
                                                         double* d_m_tmp,
                                                         double* d_TP,
                                                         bool intercept,
                                                         const double* d_mu,
                                                         int rd,
                                                         double* d_vs,
                                                         double* d_Fs,
                                                         double* d_sum_logFs,
                                                         int n_diff,
                                                         int fc_steps,
                                                         double* d_fc,
                                                         bool conf_int,
                                                         double* d_F_fc)
{
  int rd2 = rd * rd;

  // Dynamic shared memory allocation
  extern __shared__ char dyna_shared_mem[];
  double* shared_vec0  = (double*)dyna_shared_mem;
  double* shared_Z     = (double*)(dyna_shared_mem + rd * sizeof(double));
  double* shared_alpha = (double*)(dyna_shared_mem + 2 * rd * sizeof(double));
  double* shared_K     = (double*)(dyna_shared_mem + 3 * rd * sizeof(double));

  __shared__ KalmanLoopSharedMemory<GemmPolicy, GemvPolicy, double> shared_mem;

  for (int bid = blockIdx.x; bid < batch_size; bid += gridDim.x) {
    /* Load Z and alpha to shared memory */
    for (int i = threadIdx.x; i < rd; i += GemmPolicy::BlockSize) {
      shared_Z[i]     = d_Z[bid * rd + i];
      shared_alpha[i] = d_alpha[bid * rd + i];
    }

    __syncthreads();  // ensure alpha and Z are loaded before 1.

    /* Initialization */
    double mu_       = intercept ? d_mu[bid] : 0.0;
    double sum_logFs = 0.0;

    /* Kalman loop */
    for (int it = 0; it < n_obs; it++) {
      // 1.
      double vt = d_ys[bid * n_obs + it];
      if (n_diff == 0) {
        vt -= shared_alpha[0];
      } else {
        vt -= MLCommon::LinAlg::_block_dot<GemmPolicy::BlockSize, true>(
          rd, shared_Z, shared_alpha, shared_mem.reduction_storage);
        __syncthreads();  // necessary to reuse shared memory
      }
      if (threadIdx.x == 0) d_vs[bid * n_obs + it] = vt;

      // 2.
      double _F;
      if (n_diff == 0) {
        _F = (d_P + bid * rd2)[0];
      } else {
        _F = MLCommon::LinAlg::_block_xAxt<GemmPolicy::BlockSize, true, false>(
          rd, shared_Z, d_P + bid * rd2, shared_mem.reduction_storage);
        __syncthreads();  // necessary to reuse shared memory
      }
      if (threadIdx.x == 0) d_Fs[bid * n_obs + it] = _F;
      if (threadIdx.x == 0 && it >= n_diff) sum_logFs += log(_F);

      // 3. K = 1/Fs[it] * T*P*Z'
      // TP = T*P (also used later)
      MLCommon::LinAlg::_block_gemm<GemmPolicy>(false,
                                                false,
                                                rd,
                                                rd,
                                                rd,
                                                1.0,
                                                d_T + bid * rd2,
                                                d_P + bid * rd2,
                                                d_TP + bid * rd2,
                                                shared_mem.gemm_storage);
      __syncthreads();  // for consistency of TP
      // K = 1/Fs[it] * TP*Z'
      double _1_Fs = 1.0 / _F;
      if (n_diff == 0) {
        MLCommon::LinAlg::_block_ax(rd, _1_Fs, d_TP + bid * rd2, shared_K);
      } else {
        MLCommon::LinAlg::_block_gemv<GemvPolicy, false>(
          rd, rd, _1_Fs, d_TP + bid * rd2, shared_Z, shared_K, shared_mem.gemv_storage[0]);
      }

      // 4. alpha = T*alpha + K*vs[it] + c
      // vec1 = T*alpha
      MLCommon::LinAlg::_block_gemv<GemvPolicy, false>(
        rd, rd, 1.0, d_T + bid * rd2, shared_alpha, shared_vec0, shared_mem.gemv_storage[1]);
      __syncthreads();  // For consistency of K and vec1
      // alpha = vec1 + K*vs[it] + c
      for (int i = threadIdx.x; i < rd; i += GemmPolicy::BlockSize) {
        double c_       = (i == n_diff) ? mu_ : 0.0;
        shared_alpha[i] = shared_vec0[i] + vt * shared_K[i] + c_;
      }

      // 5. L = T - K * Z
      if (n_diff == 0) {
        for (int i = threadIdx.x; i < rd2; i += GemmPolicy::BlockSize) {
          double _KZ             = (i < rd) ? shared_K[i] : 0.0;
          d_m_tmp[bid * rd2 + i] = d_T[bid * rd2 + i] - _KZ;
        }
      } else {
        for (int i = threadIdx.x; i < rd2; i += GemmPolicy::BlockSize) {
          double _KZ             = shared_K[i % rd] * shared_Z[i / rd];
          d_m_tmp[bid * rd2 + i] = d_T[bid * rd2 + i] - _KZ;
        }
      }

      // 6. P = T*P*L' + R*Q*R'
      __syncthreads();  // For consistency of L
      // P = TP*L'
      MLCommon::LinAlg::_block_gemm<GemmPolicy>(false,
                                                true,
                                                rd,
                                                rd,
                                                rd,
                                                1.0,
                                                d_TP + bid * rd2,
                                                d_m_tmp + bid * rd2,
                                                d_P + bid * rd2,
                                                shared_mem.gemm_storage);
      __syncthreads();  // For consistency of P
      // P = P + R*Q*R'
      /// TODO: shared mem R instead of precomputed matrix?
      for (int i = threadIdx.x; i < rd2; i += GemmPolicy::BlockSize) {
        d_P[bid * rd2 + i] += d_RQR[bid * rd2 + i];
      }

      __syncthreads();  // necessary to reuse shared memory
    }

    /* Forecast */
    for (int it = 0; it < fc_steps; it++) {
      // pred = Z * alpha
      double pred;
      if (n_diff == 0) {
        pred = shared_alpha[0];
      } else {
        pred = MLCommon::LinAlg::_block_dot<GemmPolicy::BlockSize, false>(
          rd, shared_Z, shared_alpha, shared_mem.reduction_storage);
        __syncthreads();  // necessary to reuse shared memory
      }
      if (threadIdx.x == 0) d_fc[bid * fc_steps + it] = pred;

      // alpha = T*alpha + c
      // vec0 = T*alpha
      MLCommon::LinAlg::_block_gemv<GemvPolicy, false>(
        rd, rd, 1.0, d_T + bid * rd2, shared_alpha, shared_vec0, shared_mem.gemv_storage[0]);
      __syncthreads();  // for consistency of v_tmp + reuse of shared mem
      // alpha = vec0 + c
      for (int i = threadIdx.x; i < rd; i += GemmPolicy::BlockSize) {
        double c_       = (i == n_diff) ? mu_ : 0.0;
        shared_alpha[i] = shared_vec0[i] + c_;
      }

      double _F;
      if (conf_int) {
        if (n_diff == 0) {
          _F = d_P[bid * rd2];
        } else {
          _F = MLCommon::LinAlg::_block_xAxt<GemmPolicy::BlockSize, false, false>(
            rd, shared_Z, d_P + bid * rd2, shared_mem.reduction_storage);
          __syncthreads();  // necessary to reuse shared memory
        }

        if (threadIdx.x == 0) d_F_fc[bid * fc_steps + it] = _F;
      }

      // P = T*P*T' + R*Q*R'
      // TP = T*P
      MLCommon::LinAlg::_block_gemm<GemmPolicy>(false,
                                                false,
                                                rd,
                                                rd,
                                                rd,
                                                1.0,
                                                d_T + bid * rd2,
                                                d_P + bid * rd2,
                                                d_TP + bid * rd2,
                                                shared_mem.gemm_storage);
      __syncthreads();  // for consistency of TP
      // P = TP * T'
      MLCommon::LinAlg::_block_gemm<GemmPolicy>(false,
                                                true,
                                                rd,
                                                rd,
                                                rd,
                                                1.0,
                                                d_TP + bid * rd2,
                                                d_T + bid * rd2,
                                                d_P + bid * rd2,
                                                shared_mem.gemm_storage);
      __syncthreads();  // for consistency of P
      // P = P + R*Q*R'
      /// TODO: shared mem R instead of precomputed matrix?
      for (int i = threadIdx.x; i < rd2; i += GemmPolicy::BlockSize) {
        d_P[bid * rd2 + i] += d_RQR[bid * rd2 + i];
      }
    }

    /* Write to global mem */
    if (threadIdx.x == 0) d_sum_logFs[bid] = sum_logFs;
  }
}

/**
 * Kalman loop for large matrices (r > 8).
 *
 * @param[in]  arima_mem    Pre-allocated temporary memory
 * @param[in]  d_ys         Batched time series
 * @param[in]  nobs         Number of observation per series
 * @param[in]  T            Batched transition matrix.            (r x r)
 * @param[in]  Z            Batched "design" vector               (1 x r)
 * @param[in]  RQR          Batched R*Q*R'                        (r x r)
 * @param[in]  P            Batched P                             (r x r)
 * @param[in]  alpha        Batched state vector                  (r x 1)
 * @param[in]  intercept    Do we fit an intercept?
 * @param[in]  d_mu         Batched intercept                     (1)
 * @param[in]  rd           Dimension of the state vector
 * @param[out] d_vs         Batched residuals                     (nobs)
 * @param[out] d_Fs         Batched variance of prediction errors (nobs)
 * @param[out] d_sum_logFs  Batched sum of the logs of Fs         (1)
 * @param[in]  n_diff       d + s*D
 * @param[in]  fc_steps     Number of steps to forecast
 * @param[out] d_fc         Array to store the forecast
 * @param[in]  conf_int     Whether to compute confidence intervals
 * @param[out] d_F_fc       Batched variance of forecast errors   (fc_steps)
 */
template <typename GemmPolicy, typename GemvPolicy>
void _batched_kalman_device_loop_large(const ARIMAMemory<double>& arima_mem,
                                       const double* d_ys,
                                       int n_obs,
                                       const MLCommon::LinAlg::Batched::Matrix<double>& T,
                                       const MLCommon::LinAlg::Batched::Matrix<double>& Z,
                                       const MLCommon::LinAlg::Batched::Matrix<double>& RQR,
                                       MLCommon::LinAlg::Batched::Matrix<double>& P,
                                       MLCommon::LinAlg::Batched::Matrix<double>& alpha,
                                       bool intercept,
                                       const double* d_mu,
                                       int rd,
                                       double* d_vs,
                                       double* d_Fs,
                                       double* d_sum_logFs,
                                       int n_diff,
                                       int fc_steps   = 0,
                                       double* d_fc   = nullptr,
                                       bool conf_int  = false,
                                       double* d_F_fc = nullptr)
{
  static_assert(GemmPolicy::BlockSize == GemvPolicy::BlockSize,
                "Gemm and gemv policies: block size mismatch");

  auto stream       = T.stream();
  auto cublasHandle = T.cublasHandle();
  int batch_size    = T.batches();

  // Temporary matrices
  MLCommon::LinAlg::Batched::Matrix<double> m_tmp(rd,
                                                  rd,
                                                  batch_size,
                                                  cublasHandle,
                                                  arima_mem.m_tmp_batches,
                                                  arima_mem.m_tmp_dense,
                                                  stream,
                                                  false);
  MLCommon::LinAlg::Batched::Matrix<double> TP(
    rd, rd, batch_size, cublasHandle, arima_mem.TP_batches, arima_mem.TP_dense, stream, false);

  int grid_size          = std::min(batch_size, 65536);
  size_t shared_mem_size = 4 * rd * sizeof(double);
  _batched_kalman_device_loop_large_kernel<GemmPolicy, GemvPolicy>
    <<<grid_size, GemmPolicy::BlockSize, shared_mem_size, stream>>>(d_ys,
                                                                    batch_size,
                                                                    n_obs,
                                                                    T.raw_data(),
                                                                    Z.raw_data(),
                                                                    RQR.raw_data(),
                                                                    P.raw_data(),
                                                                    alpha.raw_data(),
                                                                    m_tmp.raw_data(),
                                                                    TP.raw_data(),
                                                                    intercept,
                                                                    d_mu,
                                                                    rd,
                                                                    d_vs,
                                                                    d_Fs,
                                                                    d_sum_logFs,
                                                                    n_diff,
                                                                    fc_steps,
                                                                    d_fc,
                                                                    conf_int,
                                                                    d_F_fc);
}

/// Wrapper around functions that execute the Kalman loop (for performance)
void batched_kalman_loop(raft::handle_t& handle,
                         const ARIMAMemory<double>& arima_mem,
                         const double* ys,
                         int nobs,
                         const MLCommon::LinAlg::Batched::Matrix<double>& T,
                         const MLCommon::LinAlg::Batched::Matrix<double>& Z,
                         const MLCommon::LinAlg::Batched::Matrix<double>& RQR,
                         MLCommon::LinAlg::Batched::Matrix<double>& P0,
                         MLCommon::LinAlg::Batched::Matrix<double>& alpha,
                         bool intercept,
                         const double* d_mu,
                         const ARIMAOrder& order,
                         double* vs,
                         double* Fs,
                         double* sum_logFs,
                         int fc_steps   = 0,
                         double* d_fc   = nullptr,
                         bool conf_int  = false,
                         double* d_F_fc = nullptr)
{
  const int batch_size = T.batches();
  auto stream          = T.stream();
  int rd               = order.rd();
  int n_diff           = order.n_diff();
  dim3 numThreadsPerBlock(32, 1);
  dim3 numBlocks(raft::ceildiv<int>(batch_size, numThreadsPerBlock.x), 1);
  if (rd <= 8) {
    switch (rd) {
      case 1:
        batched_kalman_loop_kernel<1>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(ys,
                                                         nobs,
                                                         T.raw_data(),
                                                         Z.raw_data(),
                                                         RQR.raw_data(),
                                                         P0.raw_data(),
                                                         alpha.raw_data(),
                                                         intercept,
                                                         d_mu,
                                                         batch_size,
                                                         vs,
                                                         Fs,
                                                         sum_logFs,
                                                         n_diff,
                                                         fc_steps,
                                                         d_fc,
                                                         conf_int,
                                                         d_F_fc);
        break;
      case 2:
        batched_kalman_loop_kernel<2>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(ys,
                                                         nobs,
                                                         T.raw_data(),
                                                         Z.raw_data(),
                                                         RQR.raw_data(),
                                                         P0.raw_data(),
                                                         alpha.raw_data(),
                                                         intercept,
                                                         d_mu,
                                                         batch_size,
                                                         vs,
                                                         Fs,
                                                         sum_logFs,
                                                         n_diff,
                                                         fc_steps,
                                                         d_fc,
                                                         conf_int,
                                                         d_F_fc);
        break;
      case 3:
        batched_kalman_loop_kernel<3>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(ys,
                                                         nobs,
                                                         T.raw_data(),
                                                         Z.raw_data(),
                                                         RQR.raw_data(),
                                                         P0.raw_data(),
                                                         alpha.raw_data(),
                                                         intercept,
                                                         d_mu,
                                                         batch_size,
                                                         vs,
                                                         Fs,
                                                         sum_logFs,
                                                         n_diff,
                                                         fc_steps,
                                                         d_fc,
                                                         conf_int,
                                                         d_F_fc);
        break;
      case 4:
        batched_kalman_loop_kernel<4>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(ys,
                                                         nobs,
                                                         T.raw_data(),
                                                         Z.raw_data(),
                                                         RQR.raw_data(),
                                                         P0.raw_data(),
                                                         alpha.raw_data(),
                                                         intercept,
                                                         d_mu,
                                                         batch_size,
                                                         vs,
                                                         Fs,
                                                         sum_logFs,
                                                         n_diff,
                                                         fc_steps,
                                                         d_fc,
                                                         conf_int,
                                                         d_F_fc);
        break;
      case 5:
        batched_kalman_loop_kernel<5>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(ys,
                                                         nobs,
                                                         T.raw_data(),
                                                         Z.raw_data(),
                                                         RQR.raw_data(),
                                                         P0.raw_data(),
                                                         alpha.raw_data(),
                                                         intercept,
                                                         d_mu,
                                                         batch_size,
                                                         vs,
                                                         Fs,
                                                         sum_logFs,
                                                         n_diff,
                                                         fc_steps,
                                                         d_fc,
                                                         conf_int,
                                                         d_F_fc);
        break;
      case 6:
        batched_kalman_loop_kernel<6>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(ys,
                                                         nobs,
                                                         T.raw_data(),
                                                         Z.raw_data(),
                                                         RQR.raw_data(),
                                                         P0.raw_data(),
                                                         alpha.raw_data(),
                                                         intercept,
                                                         d_mu,
                                                         batch_size,
                                                         vs,
                                                         Fs,
                                                         sum_logFs,
                                                         n_diff,
                                                         fc_steps,
                                                         d_fc,
                                                         conf_int,
                                                         d_F_fc);
        break;
      case 7:
        batched_kalman_loop_kernel<7>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(ys,
                                                         nobs,
                                                         T.raw_data(),
                                                         Z.raw_data(),
                                                         RQR.raw_data(),
                                                         P0.raw_data(),
                                                         alpha.raw_data(),
                                                         intercept,
                                                         d_mu,
                                                         batch_size,
                                                         vs,
                                                         Fs,
                                                         sum_logFs,
                                                         n_diff,
                                                         fc_steps,
                                                         d_fc,
                                                         conf_int,
                                                         d_F_fc);
        break;
      case 8:
        batched_kalman_loop_kernel<8>
          <<<numBlocks, numThreadsPerBlock, 0, stream>>>(ys,
                                                         nobs,
                                                         T.raw_data(),
                                                         Z.raw_data(),
                                                         RQR.raw_data(),
                                                         P0.raw_data(),
                                                         alpha.raw_data(),
                                                         intercept,
                                                         d_mu,
                                                         batch_size,
                                                         vs,
                                                         Fs,
                                                         sum_logFs,
                                                         n_diff,
                                                         fc_steps,
                                                         d_fc,
                                                         conf_int,
                                                         d_F_fc);
        break;
    }
    CUDA_CHECK(cudaPeekAtLastError());
  } else {
    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    if (rd <= 16) {
      if (batch_size <= 2 * num_sm) {
        using GemmPolicy = MLCommon::LinAlg::BlockGemmPolicy<1, 16, 1, 1, 16, 16>;
        using GemvPolicy = MLCommon::LinAlg::BlockGemvPolicy<16, 16>;
        _batched_kalman_device_loop_large<GemmPolicy, GemvPolicy>(arima_mem,
                                                                  ys,
                                                                  nobs,
                                                                  T,
                                                                  Z,
                                                                  RQR,
                                                                  P0,
                                                                  alpha,
                                                                  intercept,
                                                                  d_mu,
                                                                  rd,
                                                                  vs,
                                                                  Fs,
                                                                  sum_logFs,
                                                                  n_diff,
                                                                  fc_steps,
                                                                  d_fc,
                                                                  conf_int,
                                                                  d_F_fc);
      } else {
        using GemmPolicy = MLCommon::LinAlg::BlockGemmPolicy<1, 16, 1, 4, 16, 4>;
        using GemvPolicy = MLCommon::LinAlg::BlockGemvPolicy<16, 4>;
        _batched_kalman_device_loop_large<GemmPolicy, GemvPolicy>(arima_mem,
                                                                  ys,
                                                                  nobs,
                                                                  T,
                                                                  Z,
                                                                  RQR,
                                                                  P0,
                                                                  alpha,
                                                                  intercept,
                                                                  d_mu,
                                                                  rd,
                                                                  vs,
                                                                  Fs,
                                                                  sum_logFs,
                                                                  n_diff,
                                                                  fc_steps,
                                                                  d_fc,
                                                                  conf_int,
                                                                  d_F_fc);
      }
    } else if (rd <= 32) {
      if (batch_size <= 2 * num_sm) {
        using GemmPolicy = MLCommon::LinAlg::BlockGemmPolicy<1, 32, 1, 4, 32, 8>;
        using GemvPolicy = MLCommon::LinAlg::BlockGemvPolicy<32, 8>;
        _batched_kalman_device_loop_large<GemmPolicy, GemvPolicy>(arima_mem,
                                                                  ys,
                                                                  nobs,
                                                                  T,
                                                                  Z,
                                                                  RQR,
                                                                  P0,
                                                                  alpha,
                                                                  intercept,
                                                                  d_mu,
                                                                  rd,
                                                                  vs,
                                                                  Fs,
                                                                  sum_logFs,
                                                                  n_diff,
                                                                  fc_steps,
                                                                  d_fc,
                                                                  conf_int,
                                                                  d_F_fc);
      } else {
        using GemmPolicy = MLCommon::LinAlg::BlockGemmPolicy<1, 32, 1, 8, 32, 4>;
        using GemvPolicy = MLCommon::LinAlg::BlockGemvPolicy<32, 4>;
        _batched_kalman_device_loop_large<GemmPolicy, GemvPolicy>(arima_mem,
                                                                  ys,
                                                                  nobs,
                                                                  T,
                                                                  Z,
                                                                  RQR,
                                                                  P0,
                                                                  alpha,
                                                                  intercept,
                                                                  d_mu,
                                                                  rd,
                                                                  vs,
                                                                  Fs,
                                                                  sum_logFs,
                                                                  n_diff,
                                                                  fc_steps,
                                                                  d_fc,
                                                                  conf_int,
                                                                  d_F_fc);
      }
    } else if (rd > 64 && rd <= 128) {
      using GemmPolicy = MLCommon::LinAlg::BlockGemmPolicy<1, 16, 1, 16, 128, 2>;
      using GemvPolicy = MLCommon::LinAlg::BlockGemvPolicy<128, 2>;
      _batched_kalman_device_loop_large<GemmPolicy, GemvPolicy>(arima_mem,
                                                                ys,
                                                                nobs,
                                                                T,
                                                                Z,
                                                                RQR,
                                                                P0,
                                                                alpha,
                                                                intercept,
                                                                d_mu,
                                                                rd,
                                                                vs,
                                                                Fs,
                                                                sum_logFs,
                                                                n_diff,
                                                                fc_steps,
                                                                d_fc,
                                                                conf_int,
                                                                d_F_fc);
    } else {
      using GemmPolicy = MLCommon::LinAlg::BlockGemmPolicy<1, 32, 1, 16, 64, 4>;
      using GemvPolicy = MLCommon::LinAlg::BlockGemvPolicy<64, 4>;
      _batched_kalman_device_loop_large<GemmPolicy, GemvPolicy>(arima_mem,
                                                                ys,
                                                                nobs,
                                                                T,
                                                                Z,
                                                                RQR,
                                                                P0,
                                                                alpha,
                                                                intercept,
                                                                d_mu,
                                                                rd,
                                                                vs,
                                                                Fs,
                                                                sum_logFs,
                                                                n_diff,
                                                                fc_steps,
                                                                d_fc,
                                                                conf_int,
                                                                d_F_fc);
    }
  }
}

template <int NUM_THREADS>
__global__ void batched_kalman_loglike_kernel(const double* d_vs,
                                              const double* d_Fs,
                                              const double* d_sumLogFs,
                                              int nobs,
                                              int batch_size,
                                              double* d_loglike,
                                              double* d_sigma2,
                                              int n_diff,
                                              double level)
{
  using BlockReduce = cub::BlockReduce<double, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int tid           = threadIdx.x;
  int bid           = blockIdx.x;
  double bid_sigma2 = 0.0;
  for (int it = 0; it < nobs; it += NUM_THREADS) {
    // vs and Fs are in time-major order (memory layout: column major)
    int idx         = (it + tid) + bid * nobs;
    double d_vs2_Fs = 0.0;
    if (it + tid >= n_diff && it + tid < nobs) {
      double _vi = d_vs[idx];
      d_vs2_Fs   = _vi * _vi / d_Fs[idx];
    }
    __syncthreads();
    double partial_sum = BlockReduce(temp_storage).Sum(d_vs2_Fs, nobs - it);
    bid_sigma2 += partial_sum;
  }
  if (tid == 0) {
    double nobs_diff_f = static_cast<double>(nobs - n_diff);
    bid_sigma2 /= nobs_diff_f;
    if (level != 0) d_sigma2[bid] = bid_sigma2;
    d_loglike[bid] =
      -.5 * (d_sumLogFs[bid] + nobs_diff_f * bid_sigma2 + nobs_diff_f * (log(2 * M_PI)));
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
__global__ void confidence_intervals(const double* d_fc,
                                     const double* d_sigma2,
                                     double* d_lower,
                                     double* d_upper,
                                     int fc_steps,
                                     double multiplier)
{
  int idx       = blockIdx.x * fc_steps + threadIdx.x;
  double fc     = d_fc[idx];
  double margin = multiplier * sqrt(d_lower[idx] * d_sigma2[blockIdx.x]);
  d_lower[idx]  = fc - margin;
  d_upper[idx]  = fc + margin;
}

void _lyapunov_wrapper(raft::handle_t& handle,
                       const ARIMAMemory<double>& arima_mem,
                       const MLCommon::LinAlg::Batched::Matrix<double>& A,
                       MLCommon::LinAlg::Batched::Matrix<double>& Q,
                       MLCommon::LinAlg::Batched::Matrix<double>& X,
                       int r)
{
  if (r <= 5) {
    auto stream       = handle.get_stream();
    auto cublasHandle = handle.get_cublas_handle();
    int batch_size    = A.batches();
    int r2            = r * r;

    //
    // Use direct solution with Kronecker product
    //

    MLCommon::LinAlg::Batched::Matrix<double> I_m_AxA(r2,
                                                      r2,
                                                      batch_size,
                                                      cublasHandle,
                                                      arima_mem.I_m_AxA_batches,
                                                      arima_mem.I_m_AxA_dense,
                                                      stream,
                                                      false);
    MLCommon::LinAlg::Batched::Matrix<double> I_m_AxA_inv(r2,
                                                          r2,
                                                          batch_size,
                                                          cublasHandle,
                                                          arima_mem.I_m_AxA_inv_batches,
                                                          arima_mem.I_m_AxA_inv_dense,
                                                          stream,
                                                          false);

    MLCommon::LinAlg::Batched::_direct_lyapunov_helper(
      A, Q, X, I_m_AxA, I_m_AxA_inv, arima_mem.I_m_AxA_P, arima_mem.I_m_AxA_info, r);
  } else {
    // Note: the other Lyapunov solver is doing temporary mem allocations,
    // but when r > 5, allocation overhead shouldn't be a bottleneck
    X = MLCommon::LinAlg::Batched::b_lyapunov(A, Q);
  }
}

/// Internal Kalman filter implementation that assumes data exists on GPU.
void _batched_kalman_filter(raft::handle_t& handle,
                            const ARIMAMemory<double>& arima_mem,
                            const double* d_ys,
                            int nobs,
                            const ARIMAOrder& order,
                            const MLCommon::LinAlg::Batched::Matrix<double>& Zb,
                            const MLCommon::LinAlg::Batched::Matrix<double>& Tb,
                            const MLCommon::LinAlg::Batched::Matrix<double>& Rb,
                            double* d_vs,
                            double* d_Fs,
                            double* d_loglike,
                            const double* d_sigma2,
                            bool intercept,
                            const double* d_mu,
                            int fc_steps,
                            double* d_fc,
                            double level,
                            double* d_lower,
                            double* d_upper)
{
  const size_t batch_size = Zb.batches();
  auto stream             = handle.get_stream();
  auto cublasHandle       = handle.get_cublas_handle();

  auto counting = thrust::make_counting_iterator(0);

  int n_diff = order.n_diff();
  int rd     = order.rd();
  int r      = order.r();

  MLCommon::LinAlg::Batched::Matrix<double> RQb(
    rd, 1, batch_size, cublasHandle, arima_mem.RQ_batches, arima_mem.RQ_dense, stream, true);
  double* d_RQ      = RQb.raw_data();
  const double* d_R = Rb.raw_data();
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
      double sigma2 = d_sigma2[bid];
      for (int i = 0; i < rd; i++) {
        d_RQ[bid * rd + i] = d_R[bid * rd + i] * sigma2;
      }
    });
  MLCommon::LinAlg::Batched::Matrix<double> RQR(
    rd, rd, batch_size, cublasHandle, arima_mem.RQR_batches, arima_mem.RQR_dense, stream, false);
  MLCommon::LinAlg::Batched::b_gemm(false, true, rd, rd, 1, 1.0, RQb, Rb, 0.0, RQR);

  // Durbin Koopman "Time Series Analysis" pg 138
  ML::PUSH_RANGE("Init P");
  MLCommon::LinAlg::Batched::Matrix<double> P(
    rd, rd, batch_size, cublasHandle, arima_mem.P_batches, arima_mem.P_dense, stream, true);
  {
    double* d_P = P.raw_data();

    if (n_diff > 0) {
      // Initialize the diffuse part with a large variance
      /// TODO: pass this as a parameter
      constexpr double kappa = 1e6;
      thrust::for_each(
        thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
          double* b_P = d_P + rd * rd * bid;
          for (int i = 0; i < n_diff; i++) {
            b_P[(rd + 1) * i] = kappa;
          }
        });

      // Initialize the stationary part by solving a Lyapunov equation
      MLCommon::LinAlg::Batched::Matrix<double> Ts(
        r, r, batch_size, cublasHandle, arima_mem.Ts_batches, arima_mem.Ts_dense, stream, false);
      MLCommon::LinAlg::Batched::Matrix<double> RQRs(r,
                                                     r,
                                                     batch_size,
                                                     cublasHandle,
                                                     arima_mem.RQRs_batches,
                                                     arima_mem.RQRs_dense,
                                                     stream,
                                                     false);
      MLCommon::LinAlg::Batched::Matrix<double> Ps(
        r, r, batch_size, cublasHandle, arima_mem.Ps_batches, arima_mem.Ps_dense, stream, false);

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
  MLCommon::LinAlg::Batched::Matrix<double> alpha(rd,
                                                  1,
                                                  batch_size,
                                                  handle.get_cublas_handle(),
                                                  arima_mem.alpha_batches,
                                                  arima_mem.alpha_dense,
                                                  stream,
                                                  false);
  if (intercept) {
    // Compute I-T*
    MLCommon::LinAlg::Batched::Matrix<double> ImT(
      r, r, batch_size, cublasHandle, arima_mem.ImT_batches, arima_mem.ImT_dense, stream, false);
    const double* d_T = Tb.raw_data();
    double* d_ImT     = ImT.raw_data();
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        const double* b_T = d_T + rd * rd * bid;
        double* b_ImT     = d_ImT + r * r * bid;
        for (int i = 0; i < r; i++) {
          for (int j = 0; j < r; j++) {
            b_ImT[r * j + i] = (i == j ? 1.0 : 0.0) - b_T[rd * (j + n_diff) + i + n_diff];
          }
        }
      });

    // For r=1, prevent I-T from being too close to [[0]] -> no solution
    if (r == 1) {
      thrust::for_each(
        thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
          if (abs(d_ImT[bid]) < 1e-3) d_ImT[bid] = raft::signPrim(d_ImT[bid]) * 1e-3;
        });
    }

    // Compute (I-T*)^-1
    MLCommon::LinAlg::Batched::Matrix<double> ImT_inv(r,
                                                      r,
                                                      batch_size,
                                                      cublasHandle,
                                                      arima_mem.ImT_inv_batches,
                                                      arima_mem.ImT_inv_dense,
                                                      stream,
                                                      false);
    MLCommon::LinAlg::Batched::Matrix<double>::inv(
      ImT, ImT_inv, arima_mem.ImT_inv_P, arima_mem.ImT_inv_info);

    // Compute (I-T*)^-1 * c -> multiply 1st column by mu
    const double* d_ImT_inv = ImT_inv.raw_data();
    double* d_alpha         = alpha.raw_data();
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
        const double* b_ImT_inv = d_ImT_inv + r * r * bid;
        double* b_alpha         = d_alpha + rd * bid;
        double mu               = d_mu[bid];
        for (int i = 0; i < n_diff; i++) {
          b_alpha[i] = 0;
        }
        for (int i = 0; i < r; i++) {
          b_alpha[i + n_diff] = b_ImT_inv[i] * mu;
        }
      });
  } else {
    // Memset alpha to 0
    CUDA_CHECK(cudaMemsetAsync(alpha.raw_data(), 0, sizeof(double) * rd * batch_size, stream));
  }

  batched_kalman_loop(handle,
                      arima_mem,
                      d_ys,
                      nobs,
                      Tb,
                      Zb,
                      RQR,
                      P,
                      alpha,
                      intercept,
                      d_mu,
                      order,
                      d_vs,
                      d_Fs,
                      arima_mem.sumLogF_buffer,
                      fc_steps,
                      d_fc,
                      level > 0,
                      d_lower);

  // Finalize loglikelihood and prediction intervals
  constexpr int NUM_THREADS = 128;
  batched_kalman_loglike_kernel<NUM_THREADS>
    <<<batch_size, NUM_THREADS, 0, stream>>>(d_vs,
                                             d_Fs,
                                             arima_mem.sumLogF_buffer,
                                             nobs,
                                             batch_size,
                                             d_loglike,
                                             arima_mem.sigma2_buffer,
                                             n_diff,
                                             level);
  CUDA_CHECK(cudaPeekAtLastError());
  if (level > 0) {
    confidence_intervals<<<batch_size, fc_steps, 0, stream>>>(
      d_fc, arima_mem.sigma2_buffer, d_lower, d_upper, fc_steps, sqrt(2.0) * erfinv(level));
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

void init_batched_kalman_matrices(raft::handle_t& handle,
                                  const double* d_ar,
                                  const double* d_ma,
                                  const double* d_sar,
                                  const double* d_sma,
                                  int nb,
                                  const ARIMAOrder& order,
                                  int rd,
                                  double* d_Z_b,
                                  double* d_R_b,
                                  double* d_T_b)
{
  ML::PUSH_RANGE(__func__);

  auto stream = handle.get_stream();

  // Note: Z is unused yet but kept to avoid reintroducing it later when
  // adding support for exogeneous variables
  cudaMemsetAsync(d_Z_b, 0.0, rd * nb * sizeof(double), stream);
  cudaMemsetAsync(d_R_b, 0.0, rd * nb * sizeof(double), stream);
  cudaMemsetAsync(d_T_b, 0.0, rd * rd * nb * sizeof(double), stream);

  int n_diff = order.n_diff();
  int r      = order.r();

  auto counting = thrust::make_counting_iterator(0);
  auto n_theta  = order.n_theta();
  auto n_phi    = order.n_phi();
  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb, [=] __device__(int bid) {
    // See TSA pg. 54 for Z, R, T matrices

    // Z = [ 1 | 0 . . 0 1 0 . . 0 1 | 1 0 . . 0 ]
    //       d |         s*D         |     r
    for (int i = 0; i < order.d; i++)
      d_Z_b[bid * rd + i] = 1.0;
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
      d_R_b[bid * rd + n_diff + i + 1] = MLCommon::TimeSeries::reduced_polynomial<false>(
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
      batch_T[n_diff * rd + offset]                 = 1.0;
    }
    if (order.D == 2) { batch_T[(n_diff - 1) * rd + order.d] = 1.0; }
    // 3. Auto-Regressive component
    for (int i = 0; i < n_phi; i++) {
      batch_T[n_diff * (rd + 1) + i] = MLCommon::TimeSeries::reduced_polynomial<true>(
        bid, d_ar, order.p, d_sar, order.P, order.s, i + 1);
    }
    for (int i = 0; i < r - 1; i++) {
      batch_T[(n_diff + i + 1) * rd + n_diff + i] = 1.0;
    }

    // If rd=2 and phi_2=-1, I-TxT is singular
    if (rd == 2 && order.p == 2 && abs(batch_T[1] + 1) < 0.01) { batch_T[1] = -0.99; }
  });

  ML::POP_RANGE();
}

void batched_kalman_filter(raft::handle_t& handle,
                           const ARIMAMemory<double>& arima_mem,
                           const double* d_ys,
                           int nobs,
                           const ARIMAParams<double>& params,
                           const ARIMAOrder& order,
                           int batch_size,
                           double* d_loglike,
                           double* d_vs,
                           int fc_steps,
                           double* d_fc,
                           double level,
                           double* d_lower,
                           double* d_upper)
{
  ML::PUSH_RANGE(__func__);

  auto cublasHandle = handle.get_cublas_handle();
  auto stream       = handle.get_stream();

  // see (3.18) in TSA by D&K
  int rd = order.rd();

  MLCommon::LinAlg::Batched::Matrix<double> Zb(
    1, rd, batch_size, cublasHandle, arima_mem.Z_batches, arima_mem.Z_dense, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> Tb(
    rd, rd, batch_size, cublasHandle, arima_mem.T_batches, arima_mem.T_dense, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> Rb(
    rd, 1, batch_size, cublasHandle, arima_mem.R_batches, arima_mem.R_dense, stream, false);

  init_batched_kalman_matrices(handle,
                               params.ar.data(),
                               params.ma.data(),
                               params.sar.data(),
                               params.sma.data(),
                               batch_size,
                               order,
                               rd,
                               Zb.raw_data(),
                               Rb.raw_data(),
                               Tb.raw_data());

  ////////////////////////////////////////////////////////////
  // Computation

  _batched_kalman_filter(handle,
                         arima_mem,
                         d_ys,
                         nobs,
                         order,
                         Zb,
                         Tb,
                         Rb,
                         d_vs,
                         arima_mem.F_buffer,
                         d_loglike,
                         params.sigma2.data(),
                         static_cast<bool>(order.k),
                         params.mu.data(),
                         fc_steps,
                         d_fc,
                         level,
                         d_lower,
                         d_upper);

  ML::POP_RANGE();
}

void batched_jones_transform(raft::handle_t& handle,
                             const ARIMAMemory<double>& arima_mem,
                             const ARIMAOrder& order,
                             int batch_size,
                             bool isInv,
                             const double* h_params,
                             double* h_Tparams)
{
  int N                       = order.complexity();
  auto stream                 = handle.get_stream();
  double* d_params            = arima_mem.d_params;
  double* d_Tparams           = arima_mem.d_Tparams;
  ARIMAParams<double> params  = {arima_mem.params_mu.data(),
                                arima_mem.params_ar.data(),
                                arima_mem.params_ma.data(),
                                arima_mem.params_sar.data(),
                                arima_mem.params_sma.data(),
                                arima_mem.params_sigma2.data()};
  ARIMAParams<double> Tparams = {arima_mem.Tparams_mu.data(),
                                 arima_mem.Tparams_ar.data(),
                                 arima_mem.Tparams_ma.data(),
                                 arima_mem.Tparams_sar.data(),
                                 arima_mem.Tparams_sma.data(),
                                 arima_mem.Tparams_sigma2.data()};

  raft::update_device(d_params, h_params, N * batch_size, stream);

  params.unpack(order, batch_size, d_params, stream);

  MLCommon::TimeSeries::batched_jones_transform(order, batch_size, isInv, params, Tparams, stream);
  Tparams.mu = params.mu.data();

  Tparams.pack(order, batch_size, d_Tparams, stream);

  raft::update_host(h_Tparams, d_Tparams, N * batch_size, stream);
}

}  // namespace ML
