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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <common/nvtx.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "arima_helpers.cuh"
#include "batched_arima.hpp"
#include "batched_kalman.hpp"

#include <cuml/cuml.hpp>

#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/matrix_vector_op.h>
#include <metrics/batched/information_criterion.h>
#include <stats/mean.h>
#include <matrix/batched_matrix.hpp>
#include "cuda_utils.h"
#include "utils.h"

namespace ML {

/**
 * @brief Prepare data by differencing if needed (simple and/or seasonal)
 *        and removing a trend if needed
 *
 * @note: It is assumed that d + D <= 2. This is enforced on the Python side
 *
 * @param[in]  handle      cuML handle
 * @param[out] d_out       Output. Shape (n_obs - d - D*s, batch_size) (device)
 * @param[in]  d_in        Input. Shape (n_obs, batch_size) (device)
 * @param[in]  batch_size  Number of series per batch
 * @param[in]  n_obs       Number of observations per series
 * @param[in]  d           Order of simple differences (0, 1 or 1)
 * @param[in]  D           Order of seasonal differences (0, 1 or 1)
 * @param[in]  s           Seasonal period if D > 0
 * @param[in]  intercept   Whether the model fits an intercept
 * @param[in]  d_mu        Mu array if intercept > 0
 *                         Shape (batch_size,) (device)
 */
static void _prepare_data(cumlHandle& handle, double* d_out, const double* d_in,
                          int batch_size, int n_obs, int d, int D, int s,
                          int intercept = 0, const double* d_mu = nullptr) {
  const auto stream = handle.getStream();

  // Only one difference (simple or seasonal)
  if (d + D == 1) {
    int period = d ? 1 : s;
    int tpb = (n_obs - period) > 512 ? 256 : 128;  // quick heuristics
    MLCommon::Matrix::batched_diff_kernel<<<batch_size, tpb, 0, stream>>>(
      d_in, d_out, n_obs, period);
    CUDA_CHECK(cudaPeekAtLastError());
  }
  // Two differences (simple or seasonal or both)
  else if (d + D == 2) {
    int period1 = d ? 1 : s;
    int period2 = d == 2 ? 1 : s;
    int tpb = (n_obs - period1 - period2) > 512 ? 256 : 128;
    MLCommon::Matrix::
      batched_second_diff_kernel<<<batch_size, tpb, 0, stream>>>(
        d_in, d_out, n_obs, period1, period2);
    CUDA_CHECK(cudaPeekAtLastError());
  }
  // If no difference and the pointers are different, copy in to out
  else if (d + D == 0 && d_in != d_out) {
    MLCommon::copy(d_out, d_in, n_obs * batch_size, stream);
  }
  // Other cases: no difference and the pointers are the same, nothing to do

  // Remove trend in-place
  if (intercept) {
    MLCommon::LinAlg::matrixVectorOp(
      d_out, d_out, d_mu, batch_size, n_obs - d - D * s, false, true,
      [] __device__(double a, double b) { return a - b; }, stream);
  }
}

/**
 * @brief Helper function that will read in src0 if the given index is
 *        negative, src1 otherwise.
 * @note  This is useful when one array is the logical continuation of
 *        another and the index is expressed relatively to the second array.
 */
static __device__ double _select_read(const double* src0, int size0,
                                      const double* src1, int idx) {
  return idx < 0 ? src0[size0 + idx] : src1[idx];
}

/**
 * @brief Kernel to undifference the data with up to two levels of simple
 *        and/or seasonal differencing.
 * @note  One thread per series.
 */
template <bool double_diff>
static __global__ void _undiff_kernel(double* d_fc, const double* d_in,
                                      int num_steps, int batch_size, int in_ld,
                                      int n_in, int s0, int s1 = 0) {
  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (bid < batch_size) {
    double* b_fc = d_fc + bid * num_steps;
    const double* b_in = d_in + bid * in_ld;
    for (int i = 0; i < num_steps; i++) {
      if (!double_diff) {  // One simple or seasonal difference
        b_fc[i] += _select_read(b_in, n_in, b_fc, i - s0);
      } else {  // Two differences (simple, seasonal or both)
        double fc_acc = -_select_read(b_in, n_in, b_fc, i - s0 - s1);
        fc_acc += _select_read(b_in, n_in, b_fc, i - s0);
        fc_acc += _select_read(b_in, n_in, b_fc, i - s1);
        b_fc[i] += fc_acc;
      }
    }
  }
}

/**
 * @brief Finalizes a forecast by adding the trend and/or undifferencing
 *
 * @note: It is assumed that d + D <= 2. This is enforced on the Python side
 *
 * @param[in]     handle      cuML handle
 * @param[in|out] d_fc        Forecast. Shape (num_steps, batch_size) (device)
 * @param[in]     d_in        Original data. Shape (n_obs, batch_size) (device)
 * @param[in]     num_steps   Number of steps forecasted
 * @param[in]     batch_size  Number of series per batch
 * @param[in]     in_ld       Leading dimension of d_in
 * @param[in]     n_in        Number of observations/predictions in d_in
 * @param[in]     d           Order of simple differences (0, 1 or 1)
 * @param[in]     D           Order of seasonal differences (0, 1 or 1)
 * @param[in]     s           Seasonal period if D > 0
 * @param[in]     intercept   Whether the model fits an intercept
 * @param[in]     d_mu        Mu array if intercept > 0
 *                            Shape (batch_size,) (device)
 */
static void _finalize_forecast(cumlHandle& handle, double* d_fc,
                               const double* d_in, int num_steps,
                               int batch_size, int in_ld, int n_in, int d,
                               int D, int s, int intercept = 0,
                               const double* d_mu = nullptr) {
  const auto stream = handle.getStream();

  // Add the trend in-place
  if (intercept) {
    MLCommon::LinAlg::matrixVectorOp(
      d_fc, d_fc, d_mu, batch_size, num_steps, false, true,
      [] __device__(double a, double b) { return a + b; }, stream);
  }

  // Undifference
  constexpr int TPB = 64;  // One thread per series -> avoid big blocks
  if (d + D == 1) {
    _undiff_kernel<false>
      <<<MLCommon::ceildiv<int>(batch_size, TPB), TPB, 0, stream>>>(
        d_fc, d_in, num_steps, batch_size, in_ld, n_in, d ? 1 : s);
    CUDA_CHECK(cudaPeekAtLastError());
  } else if (d + D == 2) {
    _undiff_kernel<true>
      <<<MLCommon::ceildiv<int>(batch_size, TPB), TPB, 0, stream>>>(
        d_fc, d_in, num_steps, batch_size, in_ld, n_in, d ? 1 : s,
        d == 2 ? 1 : s);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

/// TODO: remove one of the two residual() versions

void residual(cumlHandle& handle, const double* d_y, int batch_size, int n_obs,
              int p, int d, int q, int P, int D, int Q, int s, int intercept,
              const double* d_params, double* d_vs, bool trans, int fc_steps,
              double* d_fc) {
  ML::PUSH_RANGE(__func__);
  std::vector<double> loglike = std::vector<double>(batch_size);
  batched_loglike(handle, d_y, batch_size, n_obs, p, d, q, P, D, Q, s,
                  intercept, d_params, loglike.data(), d_vs, trans, true,
                  fc_steps, d_fc);
  ML::POP_RANGE();
}

void residual(cumlHandle& handle, const double* d_y, int batch_size, int n_obs,
              int p, int d, int q, int P, int D, int Q, int s, int intercept,
              const double* d_mu, const double* d_ar, const double* d_ma,
              const double* d_sar, const double* d_sma, double* d_vs,
              bool trans, int fc_steps, double* d_fc) {
  ML::PUSH_RANGE(__func__);
  std::vector<double> loglike = std::vector<double>(batch_size);
  batched_loglike(handle, d_y, batch_size, n_obs, p, d, q, P, D, Q, s,
                  intercept, d_mu, d_ar, d_ma, d_sar, d_sma, loglike.data(),
                  d_vs, trans, true, fc_steps, d_fc);
  ML::POP_RANGE();
}

/// TODO: see if simplifications can be made because of the recent changes
void predict(cumlHandle& handle, const double* d_y, int batch_size, int n_obs,
             int start, int end, int p, int d, int q, int P, int D, int Q,
             int s, int intercept, const double* d_params, double* d_vs,
             double* d_y_p) {
  ML::PUSH_RANGE(__func__);
  auto allocator = handle.getDeviceAllocator();
  const auto stream = handle.getStream();

  // Unpack parameters
  double *d_mu, *d_ar, *d_ma, *d_sar, *d_sma;
  allocate_params(allocator, stream, p, q, P, Q, batch_size, &d_ar, &d_ma,
                  &d_sar, &d_sma, intercept, &d_mu);
  unpack(d_params, d_mu, d_ar, d_ma, d_sar, d_sma, batch_size, p, q, P, Q,
         intercept, stream);

  // Prepare data
  int d_sD = d + D * s;
  int ld_yprep = n_obs - d_sD;
  double* d_y_prep = (double*)allocator->allocate(
    ld_yprep * batch_size * sizeof(double), stream);
  _prepare_data(handle, d_y_prep, d_y, batch_size, n_obs, d, D, s, intercept,
                d_mu);

  // Create temporary array for the forecasts
  int num_steps = std::max(end - n_obs, 0);
  double* d_y_fc = nullptr;
  if (num_steps) {
    d_y_fc = (double*)allocator->allocate(
      num_steps * batch_size * sizeof(double), stream);
  }

  // Compute the residual and forecast - provide already prepared data and
  // extracted parameters
  residual(handle, d_y_prep, batch_size, n_obs - d_sD, p, 0, q, P, 0, Q, s, 0,
           nullptr, d_ar, d_ma, d_sar, d_sma, d_vs, false, num_steps, d_y_fc);

  auto counting = thrust::make_counting_iterator(0);
  int predict_ld = end - start;

  //
  // In-sample prediction
  //

  int p_start = std::max(start, d_sD);
  int p_end = std::min(n_obs, end);

  // The prediction loop starts by filling undefined predictions with NaN,
  // then computes the predictions from the observations and residuals
  if (start < n_obs) {
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       d_y_p[0] = 0.0;
                       for (int i = 0; i < d_sD - start; i++) {
                         d_y_p[bid * predict_ld + i] = nan("");
                       }
                       for (int i = p_start; i < p_end; i++) {
                         d_y_p[bid * predict_ld + i - start] =
                           d_y[bid * n_obs + i] -
                           d_vs[bid * ld_yprep + i - d_sD];
                       }
                     });
  }

  //
  // Finalize out-of-sample forecast and copy in-sample predictions
  //

  if (num_steps) {
    // Add trend and/or undiff
    _finalize_forecast(handle, d_y_fc, d_y, num_steps, batch_size, n_obs, n_obs,
                       d, D, s, intercept, d_mu);

    // Copy forecast in d_y_p
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       for (int i = 0; i < num_steps; i++) {
                         d_y_p[bid * predict_ld + n_obs - start + i] =
                           d_y_fc[num_steps * bid + i];
                       }
                     });

    allocator->deallocate(d_y_fc, num_steps * batch_size * sizeof(double),
                          stream);
  }

  deallocate_params(allocator, stream, p, q, P, Q, batch_size, d_ar, d_ma,
                    d_sar, d_sma, intercept, d_mu);
  allocator->deallocate(d_y_prep, ld_yprep * batch_size * sizeof(double),
                        stream);
  ML::POP_RANGE();
}

void batched_loglike(cumlHandle& handle, const double* d_y, int batch_size,
                     int n_obs, int p, int d, int q, int P, int D, int Q, int s,
                     int intercept, const double* d_mu, const double* d_ar,
                     const double* d_ma, const double* d_sar,
                     const double* d_sma, double* loglike, double* d_vs,
                     bool trans, bool host_loglike, int fc_steps,
                     double* d_fc) {
  ML::PUSH_RANGE(__func__);

  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double *d_Tar, *d_Tma, *d_Tsar, *d_Tsma;
  allocate_params(allocator, stream, p, q, P, Q, batch_size, &d_Tar, &d_Tma,
                  &d_Tsar, &d_Tsma);

  if (trans) {
    batched_jones_transform(handle, p, q, P, Q, batch_size, false, d_ar, d_ma,
                            d_sar, d_sma, d_Tar, d_Tma, d_Tsar, d_Tsma);
  } else {
    // non-transformed case: just use original parameters
    CUDA_CHECK(cudaMemcpyAsync(d_Tar, d_ar, sizeof(double) * batch_size * p,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Tma, d_ma, sizeof(double) * batch_size * q,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Tsar, d_sar, sizeof(double) * batch_size * P,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Tsma, d_sma, sizeof(double) * batch_size * Q,
                               cudaMemcpyDeviceToDevice, stream));
  }

  if (d + D + intercept == 0) {
    batched_kalman_filter(handle, d_y, n_obs, d_Tar, d_Tma, d_Tsar, d_Tsma, p,
                          q, P, Q, s, batch_size, loglike, d_vs, host_loglike,
                          false, fc_steps, d_fc);
  } else {
    double* d_y_prep = (double*)allocator->allocate(
      batch_size * (n_obs - d - s * D) * sizeof(double), stream);

    _prepare_data(handle, d_y_prep, d_y, batch_size, n_obs, d, D, s, intercept,
                  d_mu);

    batched_kalman_filter(handle, d_y_prep, n_obs - d - s * D, d_Tar, d_Tma,
                          d_Tsar, d_Tsma, p, q, P, Q, s, batch_size, loglike,
                          d_vs, host_loglike, false, fc_steps, d_fc);

    allocator->deallocate(
      d_y_prep, sizeof(double) * batch_size * (n_obs - d - s * D), stream);
  }
  deallocate_params(allocator, stream, p, q, P, Q, batch_size, d_Tar, d_Tma,
                    d_Tsar, d_Tsma);
  ML::POP_RANGE();
}

void batched_loglike(cumlHandle& handle, const double* d_y, int batch_size,
                     int n_obs, int p, int d, int q, int P, int D, int Q, int s,
                     int intercept, const double* d_params, double* loglike,
                     double* d_vs, bool trans, bool host_loglike, int fc_steps,
                     double* d_fc) {
  ML::PUSH_RANGE(__func__);

  // unpack parameters
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double *d_mu, *d_ar, *d_ma, *d_sar, *d_sma;
  allocate_params(allocator, stream, p, q, P, Q, batch_size, &d_ar, &d_ma,
                  &d_sar, &d_sma, intercept, &d_mu);
  unpack(d_params, d_mu, d_ar, d_ma, d_sar, d_sma, batch_size, p, q, P, Q,
         intercept, stream);

  batched_loglike(handle, d_y, batch_size, n_obs, p, d, q, P, D, Q, s,
                  intercept, d_mu, d_ar, d_ma, d_sar, d_sma, loglike, d_vs,
                  trans, host_loglike, fc_steps, d_fc);

  deallocate_params(allocator, stream, p, q, P, Q, batch_size, d_ar, d_ma,
                    d_sar, d_sma, intercept, d_mu);
  ML::POP_RANGE();
}

void information_criterion(cumlHandle& handle, const double* d_y,
                           int batch_size, int n_obs, int p, int d, int q,
                           int P, int D, int Q, int s, int intercept,
                           const double* d_mu, const double* d_ar,
                           const double* d_ma, const double* d_sar,
                           const double* d_sma, double* ic, int ic_type) {
  ML::PUSH_RANGE(__func__);
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_vs = (double*)allocator->allocate(
    sizeof(double) * (n_obs - d - s * D) * batch_size, stream);
  double* d_ic =
    (double*)allocator->allocate(sizeof(double) * batch_size, stream);

  /* Compute log-likelihood in d_ic */
  batched_loglike(handle, d_y, batch_size, n_obs, p, d, q, P, D, Q, s,
                  intercept, d_mu, d_ar, d_ma, d_sar, d_sma, d_ic, d_vs, false,
                  false);

  /* Compute information criterion from log-likelihood and base term */
  MLCommon::Metrics::Batched::information_criterion(
    d_ic, d_ic, static_cast<MLCommon::Metrics::IC_Type>(ic_type),
    p + q + P + Q + intercept, batch_size, n_obs - d - s * D, stream);

  /* Transfer information criterion device -> host */
  MLCommon::updateHost(ic, d_ic, batch_size, stream);

  allocator->deallocate(d_vs, sizeof(double) * (n_obs - d - s * D) * batch_size,
                        stream);
  allocator->deallocate(d_ic, sizeof(double) * batch_size, stream);
  ML::POP_RANGE();
}

/**
 * Auxiliary function of _start_params: least square approximation of an
 * ARMA model (with or without seasonality)
 * @note: in this function the non-seasonal case has s=1, not s=0!
 */
static void _arma_least_squares(
  cumlHandle& handle, double* d_ar, double* d_ma,
  const MLCommon::Matrix::BatchedMatrix<double>& bm_y, int p, int q,
  int s = 1) {
  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  int batch_size = bm_y.batches();
  int n_obs = bm_y.shape().first;

  // Initialize params
  if (p)
    CUDA_CHECK(
      cudaMemsetAsync(d_ar, 0, sizeof(double) * p * batch_size, stream));
  if (q)
    CUDA_CHECK(
      cudaMemsetAsync(d_ma, 0, sizeof(double) * q * batch_size, stream));

  int ps = p * s, qs = q * s;
  int p_ar = 2 * qs;
  int r = std::max(p_ar + qs, ps);

  if ((q && p_ar >= n_obs - p_ar) || p + q >= n_obs - r) {
    // Too few observations for the estimate, keep 0
    return;
  }

  /* Matrix formed by lag matrices of y and the residuals respectively,
   * side by side. The left side will be used to estimate AR, the right
   * side to estimate MA */
  MLCommon::Matrix::BatchedMatrix<double> bm_ls_ar_res(
    n_obs - r, p + q, batch_size, cublas_handle, allocator, stream, false);
  int ar_offset = r - ps;
  int res_offset = (ps < p_ar + qs) ? 0 : ps - p_ar - qs;

  // Get residuals from an AR(p_ar) model to estimate the MA parameters
  if (q) {
    // Create lagged y
    int ls_height = n_obs - p_ar;
    MLCommon::Matrix::BatchedMatrix<double> bm_ls =
      MLCommon::Matrix::b_lagged_mat(bm_y, p_ar);

    /* Matrix for the initial AR fit, initialized by copy of y
     * (note: this is because gels works in-place ; the matrix has larger
     *  dimensions than the actual AR fit) */
    MLCommon::Matrix::BatchedMatrix<double> bm_ar_fit =
      MLCommon::Matrix::b_2dcopy(bm_y, p_ar, 0, ls_height, 1);

    // Residual, initialized as offset y to avoid one kernel call
    MLCommon::Matrix::BatchedMatrix<double> bm_residual(
      ls_height, 1, batch_size, cublas_handle, allocator, stream, false);
    MLCommon::copy(bm_residual.raw_data(), bm_ar_fit.raw_data(),
                   ls_height * batch_size, stream);

    // Initial AR fit
    MLCommon::Matrix::b_gels(bm_ls, bm_ar_fit);

    // Compute residual (technically a gemv)
    MLCommon::Matrix::b_gemm(false, false, ls_height, 1, p_ar, -1.0, bm_ls,
                             bm_ar_fit, 1.0, bm_residual);

    // Lags of the residual
    MLCommon::Matrix::b_lagged_mat(bm_residual, bm_ls_ar_res, q, n_obs - r,
                                   res_offset, (n_obs - r) * p, s);
  }

  // Lags of y
  MLCommon::Matrix::b_lagged_mat(bm_y, bm_ls_ar_res, p, n_obs - r, ar_offset, 0,
                                 s);

  /* Initializing the vector for the ARMA fit
   * (note: also in-place as described for AR fit) */
  MLCommon::Matrix::BatchedMatrix<double> bm_arma_fit =
    MLCommon::Matrix::b_2dcopy(bm_y, r, 0, n_obs - r, 1);

  // ARMA fit
  MLCommon::Matrix::b_gels(bm_ls_ar_res, bm_arma_fit);

  /* Copy the results in the AR and MA parameters batched vectors
   * Note: calling directly the kernel as there is not yet a way to wrap
   *       existing device pointers in a batched matrix */
  if (p) {
    MLCommon::Matrix::batched_2dcopy_kernel<<<batch_size, p, 0, stream>>>(
      bm_arma_fit.raw_data(), d_ar, 0, 0, n_obs - r, 1, p, 1);
    CUDA_CHECK(cudaPeekAtLastError());
  }
  if (q) {
    MLCommon::Matrix::batched_2dcopy_kernel<<<batch_size, q, 0, stream>>>(
      bm_arma_fit.raw_data(), d_ma, p, 0, n_obs - r, 1, q, 1);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

/**
 * Auxiliary function of estimate_x0: compute the starting parameters for
 * the series pre-processed by estimate_x0
 *
 * @note: bm_y can be mutated! estimate_x0 has already created a copy.
 */
static void _start_params(cumlHandle& handle, double* d_mu, double* d_ar,
                          double* d_ma, double* d_sar, double* d_sma,
                          MLCommon::Matrix::BatchedMatrix<double>& bm_y, int p,
                          int q, int P, int Q, int s, int intercept) {
  auto stream = handle.getStream();

  int batch_size = bm_y.batches();
  int n_obs = bm_y.shape().first;

  if (intercept) {
    // Compute means and write them in mu
    MLCommon::Stats::mean(d_mu, bm_y.raw_data(), batch_size, n_obs, false,
                          false, stream);

    // Center the series around their means in-place
    MLCommon::LinAlg::matrixVectorOp(
      bm_y.raw_data(), bm_y.raw_data(), d_mu, batch_size, n_obs, false, true,
      [] __device__(double a, double b) { return a - b; }, stream);
  }

  // Estimate an ARMA fit without seasonality
  if (p + q) _arma_least_squares(handle, d_ar, d_ma, bm_y, p, q);

  // Estimate a seasonal ARMA fit independantly
  if (P + Q) _arma_least_squares(handle, d_sar, d_sma, bm_y, P, Q, s);
}

void estimate_x0(cumlHandle& handle, double* d_mu, double* d_ar, double* d_ma,
                 double* d_sar, double* d_sma, const double* d_y,
                 int batch_size, int n_obs, int p, int d, int q, int P, int D,
                 int Q, int s, int intercept) {
  ML::PUSH_RANGE(__func__);
  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  // Difference if necessary, copy otherwise
  MLCommon::Matrix::BatchedMatrix<double> bm_yd(
    n_obs - d - s * D, 1, batch_size, cublas_handle, allocator, stream, false);
  _prepare_data(handle, bm_yd.raw_data(), d_y, batch_size, n_obs, d, D, s);
  // Note: mu is not known yet! We just want to difference the data

  // Do the computation of the initial parameters
  _start_params(handle, d_mu, d_ar, d_ma, d_sar, d_sma, bm_yd, p, q, P, Q, s,
                intercept);
  ML::POP_RANGE();
}

}  // namespace ML
