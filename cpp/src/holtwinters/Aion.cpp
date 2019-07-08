/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cuda_runtime.h>

#include <iostream>
#include "Aion.hpp"
#include "aion_utils.hpp"
#include "holtwinters.hpp"

aion::AionStatus aion::AionInit() {
  aion::cublas::get_handle();
  aion::cusolver::get_handle();
  return aion::AionStatus::AION_SUCCESS;  // TODO(ahmad): check cublas
}

aion::AionStatus aion::AionDestroy() {
  aion::cublas::destroy_handle();
  aion::cusolver::destroy_handle();
  return aion::AionStatus::AION_SUCCESS;  // TODO(ahmad): check cublas
}

template <typename Dtype>
aion::AionStatus aion::AionTranspose(const Dtype *data_in, int m, int n,
                                     Dtype *data_out, ComputeMode mode) {
  if (!data_in || !data_out || n < 1 || m < 1)
    return aion::AionStatus::AION_INVALID_VALUE;

  if (mode == aion::ComputeMode::CPU) {
    return transpose_cpu(data_in, m, n, data_out);
  } else if (mode == aion::ComputeMode::GPU) {
    return transpose_gpu(data_in, m, n, data_out);
  } else {
    return aion::AionStatus::AION_INVALID_VALUE;
  }
}

aion::AionStatus aion::HoltWintersBufferSize(
  int n, int batch_size, int frequency, bool use_beta, bool use_gamma,
  int *start_leveltrend_len, int *start_season_len, int *components_len,
  int *error_len, int *leveltrend_coef_shift, int *season_coef_shift) {
  if (n <= 0 || batch_size <= 0) return aion::AionStatus::AION_INVALID_VALUE;
  if (use_gamma && frequency <= 0) return aion::AionStatus::AION_INVALID_VALUE;

  int w_len = use_gamma ? frequency : (use_beta ? 2 : 1);

  if (start_leveltrend_len) *start_leveltrend_len = batch_size;
  if (use_gamma && start_season_len) *start_season_len = frequency * batch_size;

  if (components_len) *components_len = (n - w_len) * batch_size;

  if (leveltrend_coef_shift)
    *leveltrend_coef_shift = (n - w_len - 1) * batch_size;
  if (use_gamma && season_coef_shift)
    *season_coef_shift = (n - w_len - frequency) * batch_size;

  if (error_len) *error_len = batch_size;

  return aion::AionStatus::AION_SUCCESS;
}

template <typename Dtype>
aion::AionStatus aion::HoltWintersDecompose(
  const Dtype *ts, int n, int batch_size, int frequency, Dtype *start_level,
  Dtype *start_trend, Dtype *start_season, int start_periods,
  SeasonalType seasonal, ComputeMode mode) {
  if (!ts) return aion::AionStatus::AION_INVALID_VALUE;
  if (n <= 0) return aion::AionStatus::AION_INVALID_VALUE;
  if (batch_size <= 0) return aion::AionStatus::AION_INVALID_VALUE;
  if (mode != GPU && mode != CPU) return aion::AionStatus::AION_INVALID_VALUE;
  if (seasonal != ADDITIVE && seasonal != MULTIPLICATIVE)
    return aion::AionStatus::AION_INVALID_VALUE;

  if (start_level != nullptr && start_trend == nullptr &&
      start_season == nullptr) {  // level decomposition
    cudaMemcpy(start_level, ts, sizeof(Dtype) * batch_size, cudaMemcpyDefault);
  } else if (start_level != nullptr && start_trend != nullptr &&
             start_season == nullptr) {  // trend decomposition
    if (n < 2) return aion::AionStatus::AION_INVALID_VALUE;
    cudaMemcpy(start_level, ts + batch_size, sizeof(Dtype) * batch_size,
               cudaMemcpyDefault);
    cudaMemcpy(start_trend, ts + batch_size, sizeof(Dtype) * batch_size,
               cudaMemcpyDefault);
    if (mode == aion::ComputeMode::GPU)
      cublas::axpy<Dtype>(batch_size, (Dtype)-1., ts,
                          start_trend);  // TODO(ahmad): check return value.
    else
      cblas::axpy<Dtype>(batch_size, (Dtype)-1., ts, start_trend);
  } else if (start_level != nullptr && start_trend != nullptr &&
             start_season != nullptr) {
    if (frequency < 2)  // Non-seasonal time series don't have STL decomposition
      return aion::AionStatus::AION_INVALID_VALUE;
    if (start_periods <
        2)  // We need at least two periods for STL decomposition
      return aion::AionStatus::AION_INVALID_VALUE;
    if (start_periods * frequency >
        n)  // Make sure we have enough data points to cover all start_periods
      return aion::AionStatus::AION_INVALID_VALUE;
    if (mode == GPU)
      return stl_decomposition_gpu(ts, n, batch_size, frequency, start_periods,
                                   start_level, start_trend, start_season,
                                   seasonal);
    else
      return stl_decomposition_cpu(ts, batch_size, frequency, start_periods,
                                   start_level, start_trend, start_season,
                                   seasonal);
  } else {
    return aion::AionStatus::AION_INVALID_VALUE;
  }

  return aion::AionStatus::AION_SUCCESS;
}

template <typename Dtype>
aion::AionStatus aion::HoltWintersEval(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  const Dtype *alpha, const Dtype *beta, const Dtype *gamma, Dtype *level,
  Dtype *trend, Dtype *season, Dtype *xhat, Dtype *error, SeasonalType seasonal,
  ComputeMode mode) {
  if (!ts) return aion::AionStatus::AION_INVALID_VALUE;
  if (n < 1 || batch_size < 1) return aion::AionStatus::AION_INVALID_VALUE;
  if ((!start_trend) != (!beta) || (!start_season) != (!gamma))
    return aion::AionStatus::AION_INVALID_VALUE;
  if (!alpha || !start_level) return aion::AionStatus::AION_INVALID_VALUE;
  if (start_season != nullptr && frequency < 2)
    return aion::AionStatus::AION_INVALID_VALUE;
  if (!level && !trend && !season && !xhat && !error)
    return aion::AionStatus::AION_SUCCESS;  // Nothing to do
  if (mode == CPU) {
    holtwinters_eval_cpu<Dtype>(ts, n, batch_size, frequency, start_level,
                                start_trend, start_season, alpha, beta, gamma,
                                level, trend, season, xhat, error,
                                seasonal);  // TODO(ahmad): return value
  } else if (mode == GPU) {
    holtwinters_eval_gpu<Dtype>(ts, n, batch_size, frequency, start_level,
                                start_trend, start_season, alpha, beta, gamma,
                                level, trend, season, xhat, error,
                                seasonal);  // TODO(ahmad): return value
  } else {
    return aion::AionStatus::AION_INVALID_VALUE;
  }
  return aion::AionStatus::AION_SUCCESS;
}

// TODO(ahmad): expose line search step size
// TODO(ahmad): add the dynamic step size to CPU version
// TODO(ahmad): min_error_diff is actually min_param_diff
// TODO(ahmad): add a min_error_diff criterion
// TODO(ahmad): update default optim params in the doc
// TODO(ahmad): if linesearch_iter_limit is reached, we update wrong nx values (nx values that don't minimze loss).
// change this to at least keep the old xs
template <typename Dtype>
aion::AionStatus aion::HoltWintersOptim(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta, Dtype *gamma,
  bool optim_gamma, Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
  Dtype *error, OptimCriterion *optim_result, OptimParams<Dtype> *optim_params,
  SeasonalType seasonal, ComputeMode mode) {
  // default values
  OptimParams<Dtype> optim_params_;
  optim_params_.eps = (Dtype)2.2204e-3;
  optim_params_.min_param_diff = (Dtype)1e-8;
  optim_params_.min_error_diff = (Dtype)1e-8;
  optim_params_.min_grad_norm = (Dtype)1e-4;
  optim_params_.bfgs_iter_limit = 1000;
  optim_params_.linesearch_iter_limit = 100;
  optim_params_.linesearch_tau = (Dtype)0.5;
  optim_params_.linesearch_c = (Dtype)0.8;
  optim_params_.linesearch_step_size = (Dtype)-1;

  if (optim_params) {
    if (optim_params->eps > .0) optim_params_.eps = optim_params->eps;
    if (optim_params->min_param_diff >= .0)
      optim_params_.min_param_diff = optim_params->min_param_diff;
    if (optim_params->min_error_diff >= .0)
      optim_params_.min_error_diff = optim_params->min_error_diff;
    if (optim_params->min_grad_norm >= .0)
      optim_params_.min_grad_norm = optim_params->min_grad_norm;
    if (optim_params->bfgs_iter_limit > 0)
      optim_params_.bfgs_iter_limit = optim_params->bfgs_iter_limit;
    if (optim_params->linesearch_iter_limit > 0)
      optim_params_.linesearch_iter_limit = optim_params->linesearch_iter_limit;
    if (optim_params->linesearch_tau > .0)
      optim_params_.linesearch_tau = optim_params->linesearch_tau;
    if (optim_params->linesearch_c > .0)
      optim_params_.linesearch_c = optim_params->linesearch_c;
    if (optim_params->linesearch_step_size > 0)
      optim_params_.linesearch_step_size = optim_params->linesearch_step_size;
  }

  if (!ts) return aion::AionStatus::AION_INVALID_VALUE;
  if (n < 1 || batch_size < 1) return aion::AionStatus::AION_INVALID_VALUE;
  if (!alpha || !start_level) return aion::AionStatus::AION_INVALID_VALUE;
  if ((!start_trend) != (!beta) || (!start_season) != (!gamma))
    return aion::AionStatus::AION_INVALID_VALUE;
  if (start_season && frequency < 2)
    return aion::AionStatus::AION_INVALID_VALUE;
  if (!optim_alpha && !optim_beta && !optim_gamma)
    return aion::AionStatus::AION_INVALID_VALUE;
  if ((optim_beta && !beta) || (optim_gamma && !gamma))
    return aion::AionStatus::AION_INVALID_VALUE;
  if (!alpha && !beta && !gamma & !level && !trend && !season && !xhat &&
      !error)
    return aion::AionStatus::AION_SUCCESS;  // Nothing to do

  if (mode == aion::ComputeMode::CPU) {
    holtwinters_optim_cpu<Dtype>(
      ts, n, batch_size, frequency, start_level, start_trend, start_season,
      alpha, optim_alpha, beta, optim_beta, gamma, optim_gamma, level, trend,
      season, xhat, error, optim_result, seasonal,
      optim_params_);  // TODO(ahmad): return
  } else if (mode == aion::ComputeMode::GPU) {
    holtwinters_optim_gpu<Dtype>(
      ts, n, batch_size, frequency, start_level, start_trend, start_season,
      alpha, optim_alpha, beta, optim_beta, gamma, optim_gamma, level, trend,
      season, xhat, error, optim_result, seasonal,
      optim_params_);  // TODO(ahmad): return
  } else {
    return aion::AionStatus::AION_INVALID_VALUE;
  }

  return aion::AionStatus::AION_SUCCESS;
}

template <typename Dtype>
aion::AionStatus aion::HoltWintersForecast(
  Dtype *forecast, int h, int batch_size, int frequency,
  const Dtype *level_coef, const Dtype *trend_coef, const Dtype *season_coef,
  SeasonalType seasonal, ComputeMode mode) {
  if (!forecast) return aion::AionStatus::AION_INVALID_VALUE;
  if (h < 1 || batch_size < 1) return aion::AionStatus::AION_INVALID_VALUE;
  if (!level_coef && !trend_coef && !season_coef)
    return aion::AionStatus::AION_INVALID_VALUE;
  if (season_coef && frequency < 2) return aion::AionStatus::AION_INVALID_VALUE;

  if (mode == aion::ComputeMode::CPU) {
    holtwinters_forecast_cpu<Dtype>(forecast, h, batch_size, frequency,
                                    level_coef, trend_coef, season_coef,
                                    seasonal);  // TODO(ahmad): return value
  } else if (mode == aion::ComputeMode::GPU) {
    holtwinters_forecast_gpu<Dtype>(forecast, h, batch_size, frequency,
                                    level_coef, trend_coef, season_coef,
                                    seasonal);  // TODO(ahmad): return value
  } else {
    return aion::AionStatus::AION_INVALID_VALUE;
  }

  return aion::AionStatus::AION_SUCCESS;
}

template aion::AionStatus aion::AionTranspose<float>(const float *data_in,
                                                     int m, int n,
                                                     float *data_out,
                                                     ComputeMode mode);
template aion::AionStatus aion::AionTranspose<double>(const double *data_in,
                                                      int m, int n,
                                                      double *data_out,
                                                      ComputeMode mode);

template aion::AionStatus aion::HoltWintersDecompose<float>(
  const float *ts, int n, int batch_size, int frequency, float *start_level,
  float *start_trend, float *start_season, int start_periods,
  SeasonalType seasonal, ComputeMode mode);
template aion::AionStatus aion::HoltWintersDecompose<double>(
  const double *ts, int n, int batch_size, int frequency, double *start_level,
  double *start_trend, double *start_season, int start_periods,
  SeasonalType seasonal, ComputeMode mode);

template aion::AionStatus aion::HoltWintersEval<float>(
  const float *ts, int n, int batch_size, int frequency,
  const float *start_level, const float *start_trend, const float *start_season,
  const float *alpha, const float *beta, const float *gamma, float *level,
  float *trend, float *season, float *xhat, float *error, SeasonalType seasonal,
  ComputeMode mode);
template aion::AionStatus aion::HoltWintersEval<double>(
  const double *ts, int n, int batch_size, int frequency,
  const double *start_level, const double *start_trend,
  const double *start_season, const double *alpha, const double *beta,
  const double *gamma, double *level, double *trend, double *season,
  double *xhat, double *error, SeasonalType seasonal, ComputeMode mode);

template aion::AionStatus aion::HoltWintersOptim(
  const float *ts, int n, int batch_size, int frequency,
  const float *start_level, const float *start_trend, const float *start_season,
  float *alpha, bool optim_alpha, float *beta, bool optim_beta, float *gamma,
  bool optim_gamma, float *level, float *trend, float *season, float *xhat,
  float *error, OptimCriterion *optim_result, OptimParams<float> *optim_params,
  SeasonalType seasonal, ComputeMode mode);
template aion::AionStatus aion::HoltWintersOptim(
  const double *ts, int n, int batch_size, int frequency,
  const double *start_level, const double *start_trend,
  const double *start_season, double *alpha, bool optim_alpha, double *beta,
  bool optim_beta, double *gamma, bool optim_gamma, double *level,
  double *trend, double *season, double *xhat, double *error,
  OptimCriterion *optim_result, OptimParams<double> *optim_params,
  SeasonalType seasonal, ComputeMode mode);

template aion::AionStatus aion::HoltWintersForecast<float>(
  float *forecast, int h, int batch_size, int frequency,
  const float *level_coef, const float *trend_coef, const float *season_coef,
  SeasonalType seasonal, ComputeMode mode);
template aion::AionStatus aion::HoltWintersForecast<double>(
  double *forecast, int h, int batch_size, int frequency,
  const double *level_coef, const double *trend_coef, const double *season_coef,
  SeasonalType seasonal, ComputeMode mode);
