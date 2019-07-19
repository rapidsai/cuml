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
#include "holtwinters_utils.h"
#include "linalg/transpose.h"

namespace ML {

// HW misc functions
void HWInit();
void HWDestroy();

template <typename Dtype>
void HWTranspose(const ML::cumlHandle &handle, Dtype *data_in, int m, int n,
                 Dtype *data_out);

void HoltWintersBufferSize(int n, int batch_size, int frequency, bool use_beta,
                           bool use_gamma, int *start_leveltrend_len,
                           int *start_season_len, int *components_len,
                           int *error_len, int *leveltrend_coef_shift,
                           int *season_coef_shift);

template <typename Dtype>
void HoltWintersDecompose(const ML::cumlHandle &handle, const Dtype *ts, int n,
                          int batch_size, int frequency, Dtype *start_level,
                          Dtype *start_trend, Dtype *start_season,
                          int start_periods, SeasonalType seasonal = ADDITIVE);

template <typename Dtype>
void HoltWintersOptim(const ML::cumlHandle &handle, const Dtype *ts, int n,
                      int batch_size, int frequency, const Dtype *start_level,
                      const Dtype *start_trend, const Dtype *start_season,
                      Dtype *alpha, bool optim_alpha, Dtype *beta,
                      bool optim_beta, Dtype *gamma, bool optim_gamma,
                      Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
                      Dtype *error, OptimCriterion *optim_result,
                      OptimParams<Dtype> *optim_params = nullptr,
                      SeasonalType seasonal = ADDITIVE);

template <typename Dtype>
void HoltWintersEval(const ML::cumlHandle &handle, const Dtype *ts, int n,
                     int batch_size, int frequency, const Dtype *start_level,
                     const Dtype *start_trend, const Dtype *start_season,
                     const Dtype *alpha, const Dtype *beta, const Dtype *gamma,
                     Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
                     Dtype *error, SeasonalType seasonal = ADDITIVE);

template <typename Dtype>
void HoltWintersForecast(const ML::cumlHandle &handle, Dtype *forecast, int h,
                         int batch_size, int frequency, const Dtype *level_coef,
                         const Dtype *trend_coef, const Dtype *season_coef,
                         SeasonalType seasonal = ADDITIVE);

template <typename Dtype>
void HoltWintersFit(const ML::cumlHandle &handle, int n, int batch_size,
                    int frequency, int start_periods, SeasonalType seasonal,
                    Dtype *data, Dtype *level_ptr, Dtype *trend_ptr,
                    Dtype *season_ptr, Dtype *SSE_error_ptr);

template <typename Dtype>
void HoltWintersPredict(const ML::cumlHandle &handle, int n, int batch_size,
                        int frequency, int h, SeasonalType seasonal,
                        Dtype *level_ptr, Dtype *trend_ptr, Dtype *season_ptr,
                        Dtype *forecast_ptr);

}  // namespace ML
