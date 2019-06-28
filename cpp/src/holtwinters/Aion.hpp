/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#pragma once
namespace aion {


enum AionStatus {
  AION_SUCCESS            = 0,
  AION_NOT_INITIALIZED    = 1,
  AION_INVALID_VALUE      = 2,
  AION_ALLOC_FAILED       = 3,
  AION_INTERNAL_ERROR     = 4
};

enum SeasonalType {
  ADDITIVE,
  MULTIPLICATIVE
};

enum ComputeMode {
  CPU,
  GPU
};

enum OptimCriterion {
  OPTIM_BFGS_ITER_LIMIT = 0,
  OPTIM_MIN_PARAM_DIFF  = 1,
  OPTIM_MIN_ERROR_DIFF  = 2,
  OPTIM_MIN_GRAD_NORM   = 3,
};

template <typename Dtype>
struct OptimParams {
  Dtype eps;
  Dtype min_param_diff;
  Dtype min_error_diff;
  Dtype min_grad_norm;
  int bfgs_iter_limit;
  int linesearch_iter_limit;
  Dtype linesearch_tau;
  Dtype linesearch_c;
  Dtype linesearch_step_size;
};

enum Norm {
  L0,
  L1,
  L2,
  LINF
};

// Aion misc functions
AionStatus AionInit();
AionStatus AionDestroy();

template<typename Dtype>
AionStatus AionTranspose(const Dtype *data_in, int m, int n, Dtype *data_out,
  ComputeMode mode = GPU);

AionStatus HoltWintersBufferSize(int n, int batch_size, int frequency,
  bool use_beta, bool use_gamma,
  int* start_leveltrend_len, int* start_season_len,
  int* components_len, int* error_len,
  int* leveltrend_coef_shift, int* season_coef_shift);

template<typename Dtype>
AionStatus HoltWintersDecompose(const Dtype *ts, int n, int batch_size, int frequency,
  Dtype *start_level, Dtype *start_trend, Dtype *start_season, int start_periods,
  SeasonalType seasonal = ADDITIVE, ComputeMode mode = GPU);

template<typename Dtype>
AionStatus HoltWintersOptim(const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta, Dtype *gamma, bool optim_gamma,
  Dtype *level, Dtype *trend, Dtype *season,
  Dtype *xhat, Dtype *error, OptimCriterion *optim_result, OptimParams<Dtype>* optim_params = nullptr,
  SeasonalType seasonal = ADDITIVE, ComputeMode mode = GPU);

template<typename Dtype>
AionStatus HoltWintersEval(const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  const Dtype *alpha, const Dtype *beta, const Dtype *gamma,
  Dtype *level, Dtype *trend, Dtype *season,
  Dtype *xhat, Dtype *error,
  SeasonalType seasonal = ADDITIVE, ComputeMode mode = GPU);

template<typename Dtype>
AionStatus HoltWintersForecast(Dtype *forecast, int h, int batch_size, int frequency,
  const Dtype *level_coef, const Dtype *trend_coef, const Dtype *season_coef,
  SeasonalType seasonal = ADDITIVE, ComputeMode mode = GPU);

template<typename Dtype>
AionStatus BoxCoxTransform(int n, int batchsize, Dtype* y,
  const Dtype* lambda1, const Dtype* lambda2 = nullptr, Dtype eps = .02,
  ComputeMode mode = GPU);
template<typename Dtype>
AionStatus BoxCoxTransform(int n, Dtype* y, Dtype lambda1, Dtype lambda2 = 0, Dtype eps = .02,
  ComputeMode mode = GPU);

template<typename Dtype>
AionStatus InvBoxCoxTransform(int n, int batchsize, Dtype* y,
  const Dtype* lambda1, const Dtype* lambda2 = nullptr, Dtype eps = .02,
  ComputeMode mode = GPU);
template<typename Dtype>
AionStatus InvBoxCoxTransform(int n, Dtype* y, Dtype lambda1, Dtype lambda2 = 0, Dtype eps = .02,
  ComputeMode mode = GPU);


template<typename Dtype>
AionStatus YeoJohnsonTransform(int n, int batchsize, Dtype* y, const Dtype* lambda, Dtype eps = .02,
  ComputeMode mode = GPU);
template<typename Dtype>
AionStatus YeoJohnsonTransform(int n, Dtype* y, Dtype lambda, Dtype eps = .02,
  ComputeMode mode = GPU);

template<typename Dtype>
AionStatus InvYeoJohnsonTransform(int n, int batchsize, Dtype* y, const Dtype* lambda, Dtype eps = .02,
  ComputeMode mode = GPU);
template<typename Dtype>
AionStatus InvYeoJohnsonTransform(int n, Dtype* y, Dtype lambda, Dtype eps = .02,
  ComputeMode mode = GPU);


template<typename Dtype>
AionStatus AionDiff(int n, int batch_size, const Dtype *x, Dtype *y, int lag = 1, int differences = 1,
  ComputeMode mode = GPU);

template <typename Dtype>
AionStatus AionDemean(int n, int batch_size, Dtype* x, Dtype *mean);
template <typename Dtype>
AionStatus AionNormalize(int n, int batch_size, Dtype* x, aion::Norm l, Dtype* norm = nullptr, Dtype eps = 1e-4);
template <typename Dtype>
AionStatus AionScale(int n, int batch_size, Dtype* x, const Dtype* scale = nullptr, const Dtype* bias = nullptr);


template<typename Dtype>
AionStatus ARIMAForecast(int n, int batch_size, const Dtype *ts,
  Dtype *forecast, int h, int p, int d, int q,
  const Dtype *ar_params, const Dtype *ma_params, Dtype *sigma2);

}  // namespace aion
