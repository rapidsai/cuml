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

#pragma once

#include "holtwinters_params.h"
#include "internal/hw_decompose.h"
#include "internal/hw_eval.h"
#include "internal/hw_forecast.h"
#include "internal/hw_optim.h"
#include "linalg/transpose.h"

namespace ML {

template <typename Dtype>
void HWTranspose(const ML::cumlHandle &handle, Dtype *data_in, int m, int n,
                 Dtype *data_out) {
  ASSERT(!(!data_in || !data_out || n < 1 || m < 1), "HW error in in line %d",
         __LINE__);
  const ML::cumlHandle_impl &handle_impl = handle.getImpl();
  ML::detail::streamSyncer _(handle_impl);
  cudaStream_t stream = handle_impl.getStream();
  cublasHandle_t cublas_h = handle_impl.getCublasHandle();

  MLCommon::LinAlg::transpose<Dtype>(data_in, data_out, n, m, cublas_h, stream);
}

void HoltWintersBufferSize(int n, int batch_size, int frequency, bool use_beta,
                           bool use_gamma, int *start_leveltrend_len,
                           int *start_season_len, int *components_len,
                           int *error_len, int *leveltrend_coef_shift,
                           int *season_coef_shift) {
  int w_len = use_gamma ? frequency : (use_beta ? 2 : 1);

  if (start_leveltrend_len) *start_leveltrend_len = batch_size;
  if (use_gamma && start_season_len) *start_season_len = frequency * batch_size;

  if (components_len) *components_len = (n - w_len) * batch_size;

  if (leveltrend_coef_shift)
    *leveltrend_coef_shift = (n - w_len - 1) * batch_size;
  if (use_gamma && season_coef_shift)
    *season_coef_shift = (n - w_len - frequency) * batch_size;

  if (error_len) *error_len = batch_size;
}

template <typename Dtype>
void HoltWintersDecompose(const ML::cumlHandle &handle, const Dtype *ts, int n,
                          int batch_size, int frequency, Dtype *start_level,
                          Dtype *start_trend, Dtype *start_season,
                          int start_periods, SeasonalType seasonal) {
  const ML::cumlHandle_impl &handle_impl = handle.getImpl();
  ML::detail::streamSyncer _(handle_impl);
  cudaStream_t stream = handle_impl.getStream();
  cublasHandle_t cublas_h = handle_impl.getCublasHandle();

  if (start_level != nullptr && start_trend == nullptr &&
      start_season == nullptr) {  // level decomposition
    MLCommon::copy(start_level, ts, batch_size, stream);
  } else if (start_level != nullptr && start_trend != nullptr &&
             start_season == nullptr) {  // trend decomposition
    MLCommon::copy(start_level, ts + batch_size, batch_size, stream);
    MLCommon::copy(start_trend, ts + batch_size, batch_size, stream);
    const Dtype alpha = -1.;
    CUBLAS_CHECK(MLCommon::LinAlg::cublasaxpy(cublas_h, batch_size, &alpha, ts,
                                              1, start_trend, 1, stream));
    // cublas::axpy(batch_size, (Dtype)-1., ts, start_trend);
  } else if (start_level != nullptr && start_trend != nullptr &&
             start_season != nullptr) {
    stl_decomposition_gpu(handle_impl, ts, n, batch_size, frequency,
                          start_periods, start_level, start_trend, start_season,
                          seasonal);
  }
}

template <typename Dtype>
void HoltWintersEval(const ML::cumlHandle &handle, const Dtype *ts, int n,
                     int batch_size, int frequency, const Dtype *start_level,
                     const Dtype *start_trend, const Dtype *start_season,
                     const Dtype *alpha, const Dtype *beta, const Dtype *gamma,
                     Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
                     Dtype *error, SeasonalType seasonal) {
  const ML::cumlHandle_impl &handle_impl = handle.getImpl();
  ML::detail::streamSyncer _(handle_impl);
  cudaStream_t stream = handle_impl.getStream();

  ASSERT(!((!start_trend) != (!beta) || (!start_season) != (!gamma)),
         "HW error in in line %d", __LINE__);
  ASSERT(!(!alpha || !start_level), "HW error in in line %d", __LINE__);
  ASSERT(!(start_season != nullptr && frequency < 2), "HW error in in line %d",
         __LINE__);
  if (!(!level && !trend && !season && !xhat && !error)) {
    holtwinters_eval_gpu(handle_impl, ts, n, batch_size, frequency, start_level,
                         start_trend, start_season, alpha, beta, gamma, level,
                         trend, season, xhat, error,
                         seasonal);  // TODO(ahmad): return value
  }
}

// TODO(ahmad): expose line search step size
// TODO(ahmad): add the dynamic step size to CPU version
// TODO(ahmad): min_error_diff is actually min_param_diff
// TODO(ahmad): add a min_error_diff criterion
// TODO(ahmad): update default optim params in the doc
// TODO(ahmad): if linesearch_iter_limit is reached, we update wrong nx values (nx values that don't minimze loss).
// change this to at least keep the old xs
template <typename Dtype>
void HoltWintersOptim(const ML::cumlHandle &handle, const Dtype *ts, int n,
                      int batch_size, int frequency, const Dtype *start_level,
                      const Dtype *start_trend, const Dtype *start_season,
                      Dtype *alpha, bool optim_alpha, Dtype *beta,
                      bool optim_beta, Dtype *gamma, bool optim_gamma,
                      Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
                      Dtype *error, OptimCriterion *optim_result,
                      OptimParams<Dtype> *optim_params, SeasonalType seasonal) {
  const ML::cumlHandle_impl &handle_impl = handle.getImpl();
  ML::detail::streamSyncer _(handle_impl);
  cudaStream_t stream = handle_impl.getStream();

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

  ASSERT(alpha && start_level, "HW error in in line %d", __LINE__);
  ASSERT(!((!start_trend) != (!beta) || (!start_season) != (!gamma)),
         "HW error in in line %d", __LINE__);
  ASSERT(!(start_season && frequency < 2), "HW error in in line %d", __LINE__);
  ASSERT(!(!optim_alpha && !optim_beta && !optim_gamma),
         "HW error in in line %d", __LINE__);
  ASSERT(!((optim_beta && !beta) || (optim_gamma && !gamma)),
         "HW error in in line %d", __LINE__);
  if (!(!alpha && !beta && !gamma & !level && !trend && !season && !xhat &&
        !error)) {
    holtwinters_optim_gpu(
      handle_impl, ts, n, batch_size, frequency, start_level, start_trend,
      start_season, alpha, optim_alpha, beta, optim_beta, gamma, optim_gamma,
      level, trend, season, xhat, error, optim_result, seasonal,
      optim_params_);  // TODO(ahmad): return
  }
}

template <typename Dtype>
void HoltWintersForecast(const ML::cumlHandle &handle, Dtype *forecast, int h,
                         int batch_size, int frequency, const Dtype *level_coef,
                         const Dtype *trend_coef, const Dtype *season_coef,
                         SeasonalType seasonal) {
  const ML::cumlHandle_impl &handle_impl = handle.getImpl();
  ML::detail::streamSyncer _(handle_impl);
  cudaStream_t stream = handle_impl.getStream();

  ASSERT(!(!level_coef && !trend_coef && !season_coef),
         "HW error in in line %d", __LINE__);
  ASSERT(!(season_coef && frequency < 2), "HW error in in line %d", __LINE__);
  holtwinters_forecast_gpu(handle_impl, forecast, h, batch_size, frequency,
                           level_coef, trend_coef, season_coef,
                           seasonal);  // TODO(ahmad): return value
}

template <typename Dtype>
void HoltWintersFit(const ML::cumlHandle &handle, int n, int batch_size,
                    int frequency, int start_periods, SeasonalType seasonal,
                    Dtype *data, Dtype *level_ptr, Dtype *trend_ptr,
                    Dtype *season_ptr, Dtype *SSE_error_ptr) {
  const ML::cumlHandle_impl &handle_impl = handle.getImpl();
  ML::detail::streamSyncer _(handle_impl);
  cudaStream_t stream = handle_impl.getStream();

  bool optim_alpha = true, optim_beta = true, optim_gamma = true;
  // initial values for alpha, beta and gamma
  std::vector<Dtype> alpha_h(batch_size, 0.4);
  std::vector<Dtype> beta_h(batch_size, 0.3);
  std::vector<Dtype> gamma_h(batch_size, 0.3);

  int leveltrend_seed_len, season_seed_len, components_len;
  int leveltrend_coef_offset, season_coef_offset;
  int error_len;

  HoltWintersBufferSize(
    n, batch_size, frequency, optim_beta, optim_gamma,
    &leveltrend_seed_len,     // = batch_size
    &season_seed_len,         // = frequency*batch_size
    &components_len,          // = (n-w_len)*batch_size
    &error_len,               // = batch_size
    &leveltrend_coef_offset,  // = (n-wlen-1)*batch_size (last row)
    &season_coef_offset);     // = (n-wlen-frequency)*batch_size(last freq rows)

  Dtype *dataset_d;
  Dtype *level_seed_d, *trend_seed_d = nullptr, *start_season_d = nullptr;
  Dtype *level_d, *trend_d = nullptr, *season_d = nullptr;
  Dtype *alpha_d, *beta_d = nullptr, *gamma_d = nullptr;
  Dtype *error_d;

  MLCommon::allocate(dataset_d, batch_size * n);
  MLCommon::allocate(alpha_d, batch_size);
  MLCommon::updateDevice(alpha_d, alpha_h.data(), batch_size, stream);
  MLCommon::allocate(level_seed_d, leveltrend_seed_len);
  MLCommon::allocate(level_d, components_len);

  if (optim_beta) {
    MLCommon::allocate(beta_d, batch_size);
    MLCommon::updateDevice(beta_d, beta_h.data(), batch_size, stream);
    MLCommon::allocate(trend_seed_d, leveltrend_seed_len);
    MLCommon::allocate(trend_d, components_len);
  }

  if (optim_gamma) {
    MLCommon::allocate(gamma_d, batch_size);
    MLCommon::updateDevice(gamma_d, gamma_h.data(), batch_size, stream);
    MLCommon::allocate(start_season_d, season_seed_len);
    MLCommon::allocate(season_d, components_len);
  }

  MLCommon::allocate(error_d, error_len);

  // Step 1: transpose the dataset (ML expects col major dataset)
  HWTranspose(handle, data, batch_size, n, dataset_d);

  // Step 2: Decompose dataset to get seed for level, trend and seasonal values
  HoltWintersDecompose(handle, dataset_d, n, batch_size, frequency,
                       level_seed_d, trend_seed_d, start_season_d,
                       start_periods, seasonal);

  // Step 3: Find optimal alpha, beta and gamma values (seasonal HW)
  HoltWintersOptim(handle, dataset_d, n, batch_size, frequency, level_seed_d,
                   trend_seed_d, start_season_d, alpha_d, optim_alpha, beta_d,
                   optim_beta, gamma_d, optim_gamma, level_d, trend_d, season_d,
                   (Dtype *)nullptr, error_d, (OptimCriterion *)nullptr,
                   (OptimParams<Dtype> *)nullptr, seasonal);

  //getting alpha values from Device to Host for output:
  MLCommon::updateHost(level_ptr, level_d, components_len, stream);

  //getting beta values Device to Host for output:
  MLCommon::updateHost(trend_ptr, trend_d, components_len, stream);

  //getting gamma values Device to Host for output:
  MLCommon::updateHost(season_ptr, season_d, components_len, stream);

  //getting error values Device to Host for output:
  MLCommon::updateHost(SSE_error_ptr, error_d, batch_size, stream);

  // Free the allocated memory on GPU
  CUDA_CHECK(cudaFree(dataset_d));
  CUDA_CHECK(cudaFree(level_seed_d));
  CUDA_CHECK(cudaFree(trend_seed_d));
  CUDA_CHECK(cudaFree(start_season_d));
  CUDA_CHECK(cudaFree(level_d));
  CUDA_CHECK(cudaFree(trend_d));
  CUDA_CHECK(cudaFree(season_d));
  CUDA_CHECK(cudaFree(alpha_d));
  CUDA_CHECK(cudaFree(beta_d));
  CUDA_CHECK(cudaFree(gamma_d));
  CUDA_CHECK(cudaFree(error_d));
}

template <typename Dtype>
void HoltWintersPredict(const ML::cumlHandle &handle, int n, int batch_size,
                        int frequency, int h, SeasonalType seasonal,
                        Dtype *level_ptr, Dtype *trend_ptr, Dtype *season_ptr,
                        Dtype *forecast_ptr) {
  const ML::cumlHandle_impl &handle_impl = handle.getImpl();
  ML::detail::streamSyncer _(handle_impl);
  cudaStream_t stream = handle_impl.getStream();

  bool optim_alpha = true, optim_beta = true, optim_gamma = true;

  int leveltrend_seed_len, season_seed_len, components_len;
  int leveltrend_coef_offset, season_coef_offset;
  int error_len;

  HoltWintersBufferSize(
    n, batch_size, frequency, optim_beta, optim_gamma,
    &leveltrend_seed_len,     // = batch_size
    &season_seed_len,         // = frequency*batch_size
    &components_len,          // = (n-w_len)*batch_size
    &error_len,               // = batch_size
    &leveltrend_coef_offset,  // = (n-wlen-1)*batch_size (last row)
    &season_coef_offset);     // = (n-wlen-frequency)*batch_size(last freq rows)

  Dtype *forecast_d;
  Dtype *level_d, *trend_d = nullptr, *season_d = nullptr;

  MLCommon::allocate(forecast_d, batch_size * h, stream);
  MLCommon::allocate(level_d, components_len, stream);
  MLCommon::allocate(trend_d, components_len, stream);
  MLCommon::allocate(season_d, components_len, stream);

  MLCommon::updateDevice(level_d, level_ptr, components_len, stream);
  MLCommon::updateDevice(trend_d, trend_ptr, components_len, stream);
  MLCommon::updateDevice(season_d, season_ptr, components_len, stream);

  // Step 4: Do forecast
  HoltWintersForecast(handle, forecast_d, h, batch_size, frequency,
                      level_d + leveltrend_coef_offset,
                      trend_d + leveltrend_coef_offset,
                      season_d + season_coef_offset, seasonal);

  std::vector<Dtype> forecast(batch_size * h);
  //getting forecasted values
  MLCommon::updateHost(forecast.data(), forecast_d, batch_size * h, stream);

  // Get data from 1-D column major to 1-D row major for output
  long index = 0;
  for (auto i = 0; i < batch_size; ++i) {
    for (auto j = 0; j < h; ++j)
      forecast_ptr[index++] = forecast[i + j * batch_size];
  }

  // Free the allocated memory on GPU
  CUDA_CHECK(cudaFree(forecast_d));
  CUDA_CHECK(cudaFree(level_d));
  CUDA_CHECK(cudaFree(trend_d));
  CUDA_CHECK(cudaFree(season_d));
}

}  // namespace ML