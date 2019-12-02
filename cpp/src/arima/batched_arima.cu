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

#include <cmath>
#include <cstdio>
#include <tuple>
#include <vector>

#include "batched_arima.hpp"
#include "batched_kalman.hpp"
#include "cuda_utils.h"
#include "utils.h"

#include <common/nvtx.hpp>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuml/cuml.hpp>

#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/matrix_vector_op.h>
#include <metrics/batched/information_criterion.h>
#include <stats/mean.h>
#include <matrix/batched_matrix.hpp>

namespace ML {

using std::vector;

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
    MLCommon::copy(d_out, d_in, n_obs, stream);
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
                                      int num_steps, int batch_size, int n_obs,
                                      int s0, int s1 = 0) {
  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (bid < batch_size) {
    double* b_fc = d_fc + bid * num_steps;
    const double* b_in = d_in + bid * n_obs;
    for (int i = 0; i < num_steps; i++) {
      if (!double_diff) {  // One simple or seasonal difference
        b_fc[i] += _select_read(b_in, n_obs, b_fc, i - s0);
      } else {  // Two differences (simple, seasonal or both)
        double fc_acc = _select_read(b_in, n_obs, b_fc, i - s0 - s1);
        fc_acc += _select_read(b_in, n_obs, b_fc, i - s0);
        fc_acc += _select_read(b_in, n_obs, b_fc, i - s1);
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
 * @param[in]     n_obs       Number of observations per series
 * @param[in]     d           Order of simple differences (0, 1 or 1)
 * @param[in]     D           Order of seasonal differences (0, 1 or 1)
 * @param[in]     s           Seasonal period if D > 0
 * @param[in]     intercept   Whether the model fits an intercept
 * @param[in]     d_mu        Mu array if intercept > 0
 *                            Shape (batch_size,) (device)
 */
static void _finalize_forecast(cumlHandle& handle, double* d_fc,
                               const double* d_in, int num_steps,
                               int batch_size, int n_obs, int d, int D, int s,
                               int intercept = 0,
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
        d_fc, d_in, num_steps, batch_size, n_obs, d ? 1 : s);
    CUDA_CHECK(cudaPeekAtLastError());
  } else if (d + D == 2) {
    _undiff_kernel<true>
      <<<MLCommon::ceildiv<int>(batch_size, TPB), TPB, 0, stream>>>(
        d_fc, d_in, num_steps, batch_size, n_obs, d ? 1 : s, d == 2 ? 1 : s);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

void residual(cumlHandle& handle, double* d_y, int batch_size, int n_obs, int p,
              int d, int q, int intercept, double* d_params, double* d_vs,
              bool trans) {
  ML::PUSH_RANGE(__func__);
  std::vector<double> loglike = std::vector<double>(batch_size);
  batched_loglike(handle, d_y, batch_size, n_obs, p, d, q, intercept, d_params,
                  loglike.data(), d_vs, trans);
  ML::POP_RANGE();
}

void forecast(cumlHandle& handle, int num_steps, int p, int d, int q,
              int intercept, int batch_size, int n_obs, double* d_y,
              double* d_y_prep, double* d_vs, double* d_params,
              double* d_y_fc) {
  ML::PUSH_RANGE(__func__);
  auto allocator = handle.getDeviceAllocator();
  const auto stream = handle.getStream();

  // Temporary
  int D = 0, s = 0;

  // Unpack parameters
  double *d_mu, *d_ar, *d_ma;
  if (intercept)
    d_mu = (double*)allocator->allocate(sizeof(double) * batch_size, stream);
  if (p)
    d_ar =
      (double*)allocator->allocate(sizeof(double) * batch_size * p, stream);
  if (q)
    d_ma =
      (double*)allocator->allocate(sizeof(double) * batch_size * q, stream);
  unpack(d_params, d_mu, d_ar, d_ma, batch_size, p, q, intercept, stream);

  int ld_yprep = n_obs - d - D * s;

  // Prepare data if given unprepared data
  bool dealloc_yprep = false;
  if (d_y_prep == nullptr) {
    if (intercept + d + D == 0)
      d_y_prep = d_y;
    else {
      d_y_prep = (double*)allocator->allocate(
        ld_yprep * batch_size * sizeof(double), stream);
      dealloc_yprep = true;
      _prepare_data(handle, d_y_prep, d_y, batch_size, n_obs, d, 0, 0,
                    intercept, d_mu);
    }
  }

  const auto counting = thrust::make_counting_iterator(0);

  // Copy data into temporary work arrays
  double* d_y_ =
    (double*)allocator->allocate((p + num_steps) * batch_size, stream);
  double* d_vs_ =
    (double*)allocator->allocate((q + num_steps) * batch_size, stream);
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     if (p > 0) {
                       for (int ip = 0; ip < p; ip++) {
                         d_y_[(p + num_steps) * bid + ip] =
                           d_y_prep[ld_yprep * bid + ld_yprep - p + ip];
                       }
                     }
                     if (q > 0) {
                       for (int iq = 0; iq < q; iq++) {
                         d_vs_[(q + num_steps) * bid + iq] =
                           d_vs[ld_yprep * bid + ld_yprep - q + iq];
                       }
                     }
                   });

  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     for (int i = 0; i < num_steps; i++) {
                       auto it = num_steps * bid + i;
                       d_y_fc[it] = 0.0;
                       if (p > 0) {
                         double dot_ar_y = 0.0;
                         for (int ip = 0; ip < p; ip++) {
                           dot_ar_y += d_ar[p * bid + ip] *
                                       d_y_[(p + num_steps) * bid + i + ip];
                         }
                         d_y_fc[it] += dot_ar_y;
                       }
                       if (q > 0 && i < q) {
                         double dot_ma_y = 0.0;
                         for (int iq = 0; iq < q; iq++) {
                           dot_ma_y += d_ma[q * bid + iq] *
                                       d_vs_[(q + num_steps) * bid + i + iq];
                         }
                         d_y_fc[it] += dot_ma_y;
                       }
                       if (p > 0) {
                         d_y_[(p + num_steps) * bid + i + p] = d_y_fc[it];
                       }
                     }
                   });

  _finalize_forecast(handle, d_y_fc, d_y, num_steps, batch_size, n_obs, d, D, s,
                     intercept, d_mu);

  allocator->deallocate(d_mu, sizeof(double) * batch_size, stream);
  allocator->deallocate(d_ar, sizeof(double) * p * batch_size, stream);
  allocator->deallocate(d_ma, sizeof(double) * q * batch_size, stream);
  allocator->deallocate(d_y_, (p + num_steps) * batch_size, stream);
  allocator->deallocate(d_vs_, (q + num_steps) * batch_size, stream);
  if (dealloc_yprep)
    allocator->deallocate(d_y_prep, ld_yprep * batch_size * sizeof(double),
                          stream);
  ML::POP_RANGE();
}

void predict_in_sample(cumlHandle& handle, double* d_y, int batch_size,
                       int n_obs, int p, int d, int q, int intercept,
                       double* d_params, double* d_vs, double* d_y_p) {
  ML::PUSH_RANGE(__func__);
  residual(handle, d_y, batch_size, n_obs, p, d, q, intercept, d_params, d_vs,
           false);
  auto stream = handle.getStream();
  double* d_y_diff;

  ///TODO: update for seasonality
  if (d == 0) {
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       for (int i = 0; i < n_obs; i++) {
                         int it = bid * n_obs + i;
                         d_y_p[it] = d_y[it] - d_vs[it];
                       }
                     });
  } else {
    ///TODO: compute diff with _prepare_data
    d_y_diff = (double*)handle.getDeviceAllocator()->allocate(
      sizeof(double) * batch_size * (n_obs - 1), handle.getStream());
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       for (int i = 0; i < n_obs - 1; i++) {
                         int it = bid * n_obs + i;
                         int itd = bid * (n_obs - 1) + i;
                         // note: d_y[it] + (d_y[it + 1] - d_y[it]) - d_vs[itd]
                         //    -> d_y[it+1] - d_vs[itd]
                         d_y_p[it] = d_y[it + 1] - d_vs[itd];
                         d_y_diff[itd] = d_y[it + 1] - d_y[it];
                         if (intercept)
                           d_y_diff[itd] -= d_params[(p + q + intercept) * bid];
                       }
                     });
  }

  // due to `differencing` we need to forecast a single step to make the
  // in-sample prediction the same length as the original signal.
  if (d == 1) {
    double* d_y_fc = (double*)handle.getDeviceAllocator()->allocate(
      sizeof(double) * batch_size, handle.getStream());
    forecast(handle, 1, p, d, q, intercept, batch_size, n_obs, d_y, d_y_diff,
             d_vs, d_params, d_y_fc);

    // append forecast to end of in-sample prediction
    auto counting = thrust::make_counting_iterator(0);
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       d_y_p[bid * n_obs + (n_obs - 1)] = d_y_fc[bid];
                     });
    handle.getDeviceAllocator()->deallocate(
      d_y_diff, sizeof(double) * batch_size * (n_obs - 1), handle.getStream());
    handle.getDeviceAllocator()->deallocate(d_y_fc, sizeof(double) * batch_size,
                                            handle.getStream());
  }
  ML::POP_RANGE();
}

void batched_loglike(cumlHandle& handle, double* d_y, int batch_size, int n_obs,
                     int p, int d, int q, int intercept, double* d_mu,
                     double* d_ar, double* d_ma, double* loglike, double* d_vs,
                     bool trans, bool host_loglike) {
  using std::get;

  ML::PUSH_RANGE(__func__);

  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_Tar =
    (double*)allocator->allocate(sizeof(double) * batch_size * p, stream);
  double* d_Tma =
    (double*)allocator->allocate(sizeof(double) * batch_size * q, stream);

  if (trans) {
    batched_jones_transform(handle, p, q, batch_size, false, d_ar, d_ma, d_Tar,
                            d_Tma);
  } else {
    // non-transformed case: just use original parameters
    CUDA_CHECK(cudaMemcpyAsync(d_Tar, d_ar, sizeof(double) * batch_size * p,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Tma, d_ma, sizeof(double) * batch_size * q,
                               cudaMemcpyDeviceToDevice, stream));
  }

  ///TODO: use _prepare_data
  if (d == 0) {
    // no diff
    batched_kalman_filter(handle, d_y, n_obs, d_Tar, d_Tma, p, q, batch_size,
                          loglike, d_vs);
  } else if (d == 1) {
    ////////////////////////////////////////////////////////////
    // diff and center (with `mu`):
    ////////////////////////////////////////////////////////////

    // make device array and pointer
    double* y_diff = (double*)allocator->allocate(
      batch_size * (n_obs - 1) * sizeof(double), stream);

    {
      auto counting = thrust::make_counting_iterator(0);
      // TODO: This for_each should probably go over samples, so batches
      // are in the inner loop.
      thrust::for_each(thrust::cuda::par.on(stream), counting,
                       counting + batch_size, [=] __device__(int bid) {
                         double mu_ib = d_mu[bid];
                         for (int i = 0; i < n_obs - 1; i++) {
                           // diff and center (with `mu` parameter)
                           y_diff[bid * (n_obs - 1) + i] =
                             (d_y[bid * n_obs + i + 1] - d_y[bid * n_obs + i]) -
                             mu_ib;
                         }
                       });
    }

    batched_kalman_filter(handle, y_diff, n_obs - d, d_Tar, d_Tma, p, q,
                          batch_size, loglike, d_vs);

    allocator->deallocate(y_diff, sizeof(double) * batch_size * (n_obs - 1),
                          stream);
  } else {
    throw std::runtime_error("Not supported difference parameter: d=0, 1");
  }
  allocator->deallocate(d_Tar, sizeof(double) * p * batch_size, stream);
  allocator->deallocate(d_Tma, sizeof(double) * q * batch_size, stream);
  ML::POP_RANGE();
}

void batched_loglike(cumlHandle& handle, double* d_y, int batch_size, int n_obs,
                     int p, int d, int q, int intercept, double* d_params,
                     double* loglike, double* d_vs, bool trans,
                     bool host_loglike) {
  ML::PUSH_RANGE(__func__);

  // unpack parameters
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_mu =
    (double*)allocator->allocate(sizeof(double) * batch_size, stream);
  double* d_ar =
    (double*)allocator->allocate(sizeof(double) * batch_size * p, stream);
  double* d_ma =
    (double*)allocator->allocate(sizeof(double) * batch_size * q, stream);

  // params -> (mu, ar, ma)
  unpack(d_params, d_mu, d_ar, d_ma, batch_size, p, q, intercept, stream);

  batched_loglike(handle, d_y, batch_size, n_obs, p, d, q, intercept, d_mu,
                  d_ar, d_ma, loglike, d_vs, trans, host_loglike);

  allocator->deallocate(d_mu, sizeof(double) * batch_size, stream);
  allocator->deallocate(d_ar, sizeof(double) * p * batch_size, stream);
  allocator->deallocate(d_ma, sizeof(double) * q * batch_size, stream);
  ML::POP_RANGE();
}

void information_criterion(cumlHandle& handle, double* d_y, int batch_size,
                           int n_obs, int p, int d, int q, int intercept,
                           double* d_mu, double* d_ar, double* d_ma, double* ic,
                           int ic_type) {
  ML::PUSH_RANGE(__func__);
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_vs = (double*)allocator->allocate(
    sizeof(double) * (n_obs - d) * batch_size, stream);
  double* d_ic =
    (double*)allocator->allocate(sizeof(double) * batch_size, stream);

  /* Compute log-likelihood in d_ic */
  batched_loglike(handle, d_y, batch_size, n_obs, p, d, q, intercept, d_mu,
                  d_ar, d_ma, d_ic, d_vs, true, false);

  /* Compute information criterion from log-likelihood and base term */
  MLCommon::Metrics::Batched::information_criterion(
    d_ic, d_ic, static_cast<MLCommon::Metrics::IC_Type>(ic_type),
    p + q + intercept, batch_size, n_obs, stream);

  /* Transfer information criterion device -> host */
  MLCommon::updateHost(ic, d_ic, batch_size, stream);

  allocator->deallocate(d_vs, sizeof(double) * (n_obs - d) * batch_size,
                        stream);
  allocator->deallocate(d_ic, sizeof(double) * batch_size, stream);
  ML::POP_RANGE();
}

/**
 * Auxiliary function of estimate_x0: compute the starting parameters for
 * the series pre-processed by estimate_x0
 *
 * @note: bm_y is mutated! estimate_x0 has already created a copy.
 */
static void _start_params(cumlHandle& handle, double* d_mu, double* d_ar,
                          double* d_ma,
                          MLCommon::Matrix::BatchedMatrix<double>& bm_y,
                          int batch_size, int n_obs, int p, int q,
                          int intercept) {
  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  // Initialize params
  if (p > 0)
    CUDA_CHECK(
      cudaMemsetAsync(d_ar, 0, sizeof(double) * p * batch_size, stream));
  if (q > 0)
    CUDA_CHECK(
      cudaMemsetAsync(d_ma, 0, sizeof(double) * q * batch_size, stream));

  if (intercept) {
    // Compute means and write them in mu
    MLCommon::Stats::mean(d_mu, bm_y.raw_data(), batch_size, n_obs, false,
                          false, stream);

    // Center the series around their means in-place
    MLCommon::LinAlg::matrixVectorOp(
      bm_y.raw_data(), bm_y.raw_data(), d_mu, batch_size, n_obs, false, true,
      [] __device__(double a, double b) { return a - b; }, stream);
  }

  if (p == 0 && q == 0) {
    return;
  } else if (p != 0) {
    /* Note: p_lags fixed to p to avoid non-full-rank matrix issues */
    int p_lags = p;

    // Create lagged y
    int ls_height = n_obs - p_lags;
    MLCommon::Matrix::BatchedMatrix<double> bm_ls =
      MLCommon::Matrix::b_lagged_mat(bm_y, p_lags);

    /* Matrix for the initial AR fit, initialized by copy of y
     * (note: this is because gels works in-place ; the matrix has larger
     *  dimensions than the actual AR fit) */
    MLCommon::Matrix::BatchedMatrix<double> bm_ar_fit =
      MLCommon::Matrix::b_2dcopy(bm_y, p_lags, 0, ls_height, 1);

    // Residual if q != 0, initialized as offset y to avoid one kernel call
    MLCommon::Matrix::BatchedMatrix<double> bm_residual(
      q != 0 ? ls_height : 1, 1, batch_size, cublas_handle, allocator, stream,
      false);
    if (q != 0) {
      MLCommon::copy(bm_residual.raw_data(), bm_ar_fit.raw_data(),
                     ls_height * batch_size, stream);
    }

    // Initial AR fit
    MLCommon::Matrix::b_gels(bm_ls, bm_ar_fit);

    /* If q == 0, stop here and use these parameters
     * Note: if q == 0, we must always choose p_lags == p! */
    if (q == 0) {
      MLCommon::Matrix::batched_2dcopy_kernel<<<batch_size, p, 0, stream>>>(
        bm_ar_fit.raw_data(), d_ar, 0, 0, ls_height, 1, p, 1);
      CUDA_CHECK(cudaPeekAtLastError());
    } else {
      // Compute residual (technically a gemv)
      MLCommon::Matrix::b_gemm(false, false, ls_height, 1, p_lags, -1.0, bm_ls,
                               bm_ar_fit, 1.0, bm_residual);

      /* Create matrices made of the concatenation of lagged sets of y and the
       * residual respectively, side by side.
       * Basically the left side is for AR, the right side is for MA */
      int arma_fit_offset = std::max(p_lags + q, p);
      int ls_ar_res_height = n_obs - arma_fit_offset;
      int ar_offset = (p < p_lags + q) ? (p_lags + q - p) : 0;
      int res_offset = (p < p_lags + q) ? 0 : p - p_lags - q;
      MLCommon::Matrix::BatchedMatrix<double> bm_ls_ar_res(
        ls_ar_res_height, p + q, batch_size, cublas_handle, allocator, stream,
        false);
      MLCommon::Matrix::b_lagged_mat(bm_y, bm_ls_ar_res, p, ls_ar_res_height,
                                     ar_offset, 0);
      MLCommon::Matrix::b_lagged_mat(bm_residual, bm_ls_ar_res, q,
                                     ls_ar_res_height, res_offset,
                                     ls_ar_res_height * p);

      /* Initializing the vector for the ARMA fit
       * (note: also in-place as described for AR fit) */
      MLCommon::Matrix::BatchedMatrix<double> bm_arma_fit =
        MLCommon::Matrix::b_2dcopy(bm_y, arma_fit_offset, 0, ls_ar_res_height,
                                   1);

      // ARMA fit
      MLCommon::Matrix::b_gels(bm_ls_ar_res, bm_arma_fit);

      /* Copy the results in the AR and MA parameters batched vectors
       * Note: calling directly the kernel as there is not yet a way to wrap
       *       existing device pointers in a batched matrix */
      MLCommon::Matrix::batched_2dcopy_kernel<<<batch_size, p, 0, stream>>>(
        bm_arma_fit.raw_data(), d_ar, 0, 0, ls_ar_res_height, 1, p, 1);
      CUDA_CHECK(cudaPeekAtLastError());
      MLCommon::Matrix::batched_2dcopy_kernel<<<batch_size, q, 0, stream>>>(
        bm_arma_fit.raw_data(), d_ma, p, 0, ls_ar_res_height, 1, q, 1);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  } else {  // p == 0 && q > 0
    ///TODO: `statsmodels` has a more clever estimate for this case

    // Set MA params to -1
    thrust::device_ptr<double> __ma = thrust::device_pointer_cast(d_ma);
    thrust::fill(thrust::cuda::par.on(stream), __ma, __ma + q * batch_size,
                 -1.0);
  }
}

void estimate_x0(cumlHandle& handle, double* d_mu, double* d_ar, double* d_ma,
                 const double* d_y, int batch_size, int n_obs, int p, int d,
                 int q, int intercept) {
  ML::PUSH_RANGE(__func__);
  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  /* Based on d, differenciate the series or simply copy it
   * Note: the copy is needed because _start_params writes in it */
  int actual_n_obs = n_obs - d;
  MLCommon::Matrix::BatchedMatrix<double> bm_yd(
    actual_n_obs, 1, batch_size, cublas_handle, allocator, stream, false);
  if (d == 0) {
    MLCommon::copy(bm_yd.raw_data(), d_y, n_obs * batch_size, stream);
  } else if (d == 1) {
    const int TPB = (n_obs - 1) > 512 ? 256 : 128;  // Quick heuristics
    MLCommon::Matrix::batched_diff_kernel<<<batch_size, TPB, 0, stream>>>(
      d_y, bm_yd.raw_data(), n_obs);
    CUDA_CHECK(cudaPeekAtLastError());
  } else {
    throw std::runtime_error(
      "Not supported difference parameter. Required: d=0 or 1");
  }

  // Do the computation of the initial parameters
  _start_params(handle, d_mu, d_ar, d_ma, bm_yd, batch_size, actual_n_obs, p, q,
                intercept);
  ML::POP_RANGE();
}

}  // namespace ML
