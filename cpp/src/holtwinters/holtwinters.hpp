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
#include <cuda_runtime.h>
#include <vector>
#include "Aion.hpp"

#define STMP_EPS (1e-6)
#define GOLD \
  0.381966011250105151795413165634361882279690820194237137864551377294739537181097550292792795810608862515245
#define PG_EPS 1e-10
#define SUBSTITUTE(a, b, c, d) \
  (a) = (b);                   \
  (c) = (d);

//holtwinters_decompose.hpp
template <typename Dtype>
aion::AionStatus stl_decomposition_cpu(const Dtype *ts, int batch_size,
                                       int frequency, int start_periods,
                                       Dtype *level, Dtype *trend,
                                       Dtype *season,
                                       aion::SeasonalType seasonal);
template <typename Dtype>
aion::AionStatus stl_decomposition_gpu(const Dtype *ts, int n, int batch_size,
                                       int frequency, int start_periods,
                                       Dtype *level, Dtype *trend,
                                       Dtype *season,
                                       aion::SeasonalType seasonal);

//holtwinters_eval.hpp
template <typename Dtype>
void holtwinters_eval_cpu(const Dtype *ts, int n, int batch_size, int frequency,
                          const Dtype *start_level, const Dtype *start_trend,
                          const Dtype *start_season, const Dtype *alpha,
                          const Dtype *beta, const Dtype *gamma, Dtype *level,
                          Dtype *trend, Dtype *season, Dtype *xhat,
                          Dtype *error, aion::SeasonalType seasonal);
template <typename Dtype>
void holtwinters_eval_gpu(const Dtype *ts, int n, int batch_size, int frequency,
                          const Dtype *start_level, const Dtype *start_trend,
                          const Dtype *start_season, const Dtype *alpha,
                          const Dtype *beta, const Dtype *gamma, Dtype *level,
                          Dtype *trend, Dtype *season, Dtype *xhat,
                          Dtype *error, aion::SeasonalType seasonal);

template <typename Dtype>
Dtype holtwinters_eval_host(int id, const Dtype *ts, int n, int batch_size,
                            int frequency, int shift, Dtype plevel,
                            Dtype ptrend, const Dtype *start_season,
                            const Dtype *beta, const Dtype *gamma, Dtype alpha_,
                            Dtype beta_, Dtype gamma_, Dtype *level,
                            Dtype *trend, Dtype *season, Dtype *xhat,
                            aion::SeasonalType seasonal);
template <typename Dtype, bool additive_seasonal>
__device__ Dtype holtwinters_eval_device(
  int tid, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, Dtype *pseason, int pseason_width,
  const Dtype *start_season, const Dtype *beta, const Dtype *gamma,
  Dtype alpha_, Dtype beta_, Dtype gamma_, Dtype *level, Dtype *trend,
  Dtype *season, Dtype *xhat);

template <typename Dtype>
Dtype bound_host(Dtype val, Dtype min = .0, Dtype max = 1.);

template <typename Dtype>
__device__ Dtype abs_device(Dtype val);
template <typename Dtype>
__device__ Dtype bound_device(Dtype val, Dtype min = .0, Dtype max = 1.);

//holtwinters_forecast.hpp
template <typename Dtype>
void holtwinters_forecast_cpu(Dtype *forecast, int h, int batch_size,
                              int frequency, const Dtype *level_coef,
                              const Dtype *trend_coef, const Dtype *season_coef,
                              aion::SeasonalType seasonal = aion::ADDITIVE);
template <typename Dtype>
void holtwinters_forecast_gpu(Dtype *forecast, int h, int batch_size,
                              int frequency, const Dtype *level_coef,
                              const Dtype *trend_coef, const Dtype *season_coef,
                              aion::SeasonalType seasonal = aion::ADDITIVE);

//holtwinters_optim.hpp
template <typename Dtype>
void holtwinters_optim_cpu(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta, Dtype *gamma,
  bool optim_gamma, Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
  Dtype *error, aion::OptimCriterion *optim_result, aion::SeasonalType seasonal,
  const aion::OptimParams<Dtype> optim_params);
template <typename Dtype>
void holtwinters_optim_gpu(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta, Dtype *gamma,
  bool optim_gamma, Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
  Dtype *error, aion::OptimCriterion *optim_result, aion::SeasonalType seasonal,
  const aion::OptimParams<Dtype> optim_params);

template <typename Dtype, bool additive_seasonal>
__device__ void holtwinters_finite_gradient_device(
  int tid, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, Dtype *pseason, int pseason_width,
  const Dtype *start_season, const Dtype *beta, const Dtype *gamma,
  Dtype alpha_, Dtype beta_, Dtype gamma_, Dtype *g_alpha, Dtype *g_beta,
  Dtype *g_gamma, Dtype eps = 2.2204e-6);
template <typename Dtype>
void holtwinters_finite_gradient_host(
  int id, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, const Dtype *start_season, const Dtype *beta,
  const Dtype *gamma, Dtype alpha_, Dtype beta_, Dtype gamma_, Dtype *g_alpha,
  Dtype *g_beta, Dtype *g_gamma, aion::SeasonalType seasonal,
  Dtype eps = 2.2204e-6);
