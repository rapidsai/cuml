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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "Aion.hpp"

#include "holtwinters.hpp"

#include "aion_utils.hpp"

#define IDX(n, m, N) (n + (m) * (N))

// TODO(ahmad): Fix (refactor) some len parameters.

template <typename Dtype>
void batched_ls(const std::vector<Dtype> &data, int trend_len, int batch_size,
                Dtype *level, Dtype *trend);
template <typename Dtype>
void batched_ls_solver(const std::vector<Dtype> &B,
                       const std::vector<Dtype> &rq, int batch_size, int len,
                       Dtype *level, Dtype *trend);

template <typename Dtype>
aion::AionStatus stl_decomposition_cpu(const Dtype *ts, int batch_size,
                                       int frequency, int start_periods,
                                       Dtype *start_level, Dtype *start_trend,
                                       Dtype *start_season,
                                       aion::SeasonalType seasonal) {
  const int end = start_periods * frequency;
  const int filter_size = (frequency / 2) * 2 + 1;
  const int trend_len = end - filter_size + 1;

  // Set filter
  std::vector<Dtype> filter(filter_size, 1. / frequency);
  if (frequency % 2 == 0) {
    filter.front() /= 2;
    filter.back() /= 2;
  }

  // Set trend
  std::vector<Dtype> trend(batch_size * trend_len, 0.);
  for (int j = 0; j < batch_size; ++j) {
    for (int i = 0; i < trend_len; ++i) {
      for (int k = 0; k < filter_size; ++k) {
        trend[i * batch_size + j] += filter[k] * ts[(i + k) * batch_size + j];
      }
    }
  }

  // Calculate seasonal elements
  std::vector<Dtype> season(batch_size * trend_len);
  for (int j = 0; j < batch_size; ++j) {
    for (int i = 0; i < trend_len; ++i) {
      const int sidx = i * batch_size + j;
      const int tsidx = (i + filter_size / 2) * batch_size + j;
      if (seasonal == aion::SeasonalType::ADDITIVE)
        season[sidx] = ts[tsidx] - trend[sidx];
      else if (trend[sidx] != 0.)
        season[sidx] = ts[tsidx] / trend[sidx];
      else
        season[sidx] = ts[tsidx];
    }
  }

  // Average seasons
  const int half_filter_size = filter_size / 2;
  memset(start_season, 0, frequency * batch_size * sizeof(Dtype));
  for (int j = 0; j < batch_size; ++j) {
    Dtype season_mean = 0.0;
    for (int i = 0; i < frequency; ++i) {
      int counter = 0;
      for (int k = i - half_filter_size; k < trend_len; k = k + frequency) {
        if (k >= 0) {
          start_season[i * batch_size + j] += season[k * batch_size + j];
          ++counter;
        }
      }
      start_season[i * batch_size + j] /= counter;
      season_mean += start_season[i * batch_size + j];
    }
    season_mean /= frequency;
    for (int i = 0; i < frequency; ++i) {
      if (seasonal == aion::SeasonalType::ADDITIVE)
        start_season[i * batch_size + j] =
          start_season[i * batch_size + j] - season_mean;
      else if (season_mean != 0.)
        start_season[i * batch_size + j] =
          start_season[i * batch_size + j] / season_mean;
    }
  }

  // Solve lease squares to get level and trend initial values
  batched_ls<Dtype>(trend, trend_len, batch_size, start_level, start_trend);

  return aion::AION_SUCCESS;
}

template <typename Dtype>
void batched_ls(const std::vector<Dtype> &data, int trend_len, int batch_size,
                Dtype *level, Dtype *trend) {
  // QR decomposition of A
  std::vector<Dtype> A(2 * trend_len);
  std::vector<Dtype> tau(2);
  for (int i = 0; i < trend_len; ++i) {
    A[2 * i] = (Dtype)1.;
    A[2 * i + 1] = (Dtype)(i + 1);
  }
  aion::lapack::geqrf<Dtype>(trend_len, 2, A.data(), 2,
                             tau.data());  // TODO(ahmad): return value

  // Inverse of R (2x2 upper triangular matrix)
  Dtype R_inv[4];
  Dtype a = A[0], b = A[1], d = A[3];
  Dtype factor = 1. / (a * d);
  R_inv[0] = factor * d;
  R_inv[1] = -factor * b;
  R_inv[2] = 0.;
  R_inv[3] = factor * a;

  // R1QT = inv(R)*transpose(Q)
  std::vector<Dtype> R1Qt(2 * trend_len);
  aion::lapack::orgqr<Dtype>(trend_len, 2, 2, A.data(), 2, tau.data());
  aion::cblas::gemm<Dtype>(CblasNoTrans, CblasTrans, 2, trend_len, 2, 1., R_inv,
                           2, A.data(), 2, 0., R1Qt.data(), trend_len);

  batched_ls_solver<Dtype>(data, R1Qt, batch_size, trend_len, level, trend);
}

template <typename Dtype>
void batched_ls_solver(const std::vector<Dtype> &B,
                       const std::vector<Dtype> &rq, int batch_size, int len,
                       Dtype *level, Dtype *trend) {
  std::memset(level, 0, batch_size * sizeof(Dtype));
  std::memset(trend, 0, batch_size * sizeof(Dtype));
  for (int i = 0; i < len; ++i) {
    Dtype rq_level = rq[i];
    Dtype rq_trend = rq[len + i];
    for (int j = 0; j < batch_size; ++j) {
      Dtype b = B[j + i * batch_size];
      level[j] += rq_level * b;
      trend[j] += rq_trend * b;
    }
  }
}

template aion::AionStatus stl_decomposition_cpu<float>(
  const float *ts, int batch_size, int frequency, int start_periods,
  float *level, float *trend, float *season, aion::SeasonalType seasonal);
template aion::AionStatus stl_decomposition_cpu<double>(
  const double *ts, int batch_size, int frequency, int start_periods,
  double *level, double *trend, double *season, aion::SeasonalType seasonal);

template <typename Dtype>
Dtype bound_host(Dtype val, Dtype min, Dtype max) {
  return std::min(std::max(val, min), max);
}

template <typename Dtype>
void holtwinters_eval_cpu(const Dtype *ts, int n, int batch_size, int frequency,
                          const Dtype *start_level, const Dtype *start_trend,
                          const Dtype *start_season, const Dtype *alpha,
                          const Dtype *beta, const Dtype *gamma, Dtype *level,
                          Dtype *trend, Dtype *season, Dtype *xhat,
                          Dtype *error, aion::SeasonalType seasonal) {
  for (int j = 0; j < batch_size; ++j) {
    int shift = 1;
    Dtype plevel = start_level[j], ptrend = .0;
    Dtype alpha_ = alpha[j];
    Dtype beta_ = beta ? beta[j] : .0;
    Dtype gamma_ = gamma ? gamma[j] : .0;

    if (gamma) {
      shift = frequency;
      ptrend = beta ? start_trend[j] : .0;
    } else if (beta) {
      shift = 2;
      ptrend = start_trend[j];
    }

    Dtype error_ = holtwinters_eval_host<Dtype>(
      j, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
      beta, gamma, alpha_, beta_, gamma_, level, trend, season, xhat, seasonal);

    if (error) error[j] = error_;
  }
}

template <typename Dtype>
Dtype holtwinters_eval_host(int id, const Dtype *ts, int n, int batch_size,
                            int frequency, int shift, Dtype plevel,
                            Dtype ptrend, const Dtype *start_season,
                            const Dtype *beta, const Dtype *gamma, Dtype alpha_,
                            Dtype beta_, Dtype gamma_, Dtype *level,
                            Dtype *trend, Dtype *season, Dtype *xhat,
                            aion::SeasonalType seasonal) {
  alpha_ = bound_host(alpha_);
  beta_ = bound_host(beta_);
  gamma_ = bound_host(gamma_);
  std::vector<Dtype> pseason;
  if (gamma) pseason.assign(frequency, .0);

  Dtype error_ = .0;
  Dtype clevel = .0, ctrend = .0, cseason = .0;
  for (int i = 0; i < n - shift; i++) {
    int s = i % frequency;
    Dtype pts = ts[IDX(id, i + shift, batch_size)];
    Dtype leveltrend = plevel + ptrend;

    // xhat
    Dtype stmp;
    if (gamma)
      stmp = i < frequency ? start_season[IDX(id, i, batch_size)] : pseason[s];
    else
      stmp = static_cast<Dtype>(seasonal == aion::SeasonalType::MULTIPLICATIVE);
    Dtype xhat_ = leveltrend;
    if (seasonal == aion::SeasonalType::ADDITIVE)
      xhat_ += stmp;
    else
      xhat_ *= stmp;

    // Error
    Dtype diff = pts - xhat_;
    error_ += diff * diff;

    // Level
    if (seasonal == aion::SeasonalType::ADDITIVE) {
      clevel = alpha_ * (pts - stmp) + (1 - alpha_) * (leveltrend);
    } else {
      Dtype stmp_eps = std::fabs(stmp) > STMP_EPS ? stmp : STMP_EPS;
      clevel = alpha_ * (pts / stmp_eps) + (1 - alpha_) * (leveltrend);
    }

    // Trend
    if (beta) {
      ctrend = beta_ * (clevel - plevel) + (1 - beta_) * ptrend;
      ptrend = ctrend;
    }

    // Seasonal
    if (gamma) {
      if (seasonal == aion::SeasonalType::ADDITIVE)
        cseason = gamma_ * (pts - clevel) + (1 - gamma_) * stmp;
      else
        cseason = gamma_ * (pts / clevel) + (1 - gamma_) * stmp;
      pseason[s] = cseason;
    }

    plevel = clevel;

    if (level) level[IDX(id, i, batch_size)] = clevel;
    if (trend) trend[IDX(id, i, batch_size)] = ctrend;
    if (season) season[IDX(id, i, batch_size)] = cseason;
    if (xhat) xhat[IDX(id, i, batch_size)] = xhat_;
  }
  return error_;
}

template void holtwinters_eval_cpu<float>(
  const float *ts, int n, int batch_size, int frequency, const float *alpha,
  const float *beta, const float *gamma, const float *start_level,
  const float *start_trend, const float *start_season, float *level,
  float *trend, float *season, float *xhat, float *error,
  aion::SeasonalType seasonal);
template void holtwinters_eval_cpu<double>(
  const double *ts, int n, int batch_size, int frequency, const double *alpha,
  const double *beta, const double *gamma, const double *start_level,
  const double *start_trend, const double *start_season, double *level,
  double *trend, double *season, double *xhat, double *error,
  aion::SeasonalType seasonal);

template float bound_host<float>(float val, float min, float max);
template double bound_host<double>(double val, double min, double max);

template float holtwinters_eval_host<float>(
  int id, const float *ts, int n, int batch_size, int frequency, int shift,
  float plevel, float ptrend, const float *start_season, const float *beta,
  const float *gamma, float alpha_, float beta_, float gamma_, float *level,
  float *trend, float *season, float *xhat, aion::SeasonalType seasonal);
template double holtwinters_eval_host<double>(
  int id, const double *ts, int n, int batch_size, int frequency, int shift,
  double plevel, double ptrend, const double *start_season, const double *beta,
  const double *gamma, double alpha_, double beta_, double gamma_,
  double *level, double *trend, double *season, double *xhat,
  aion::SeasonalType seasonal);

template <typename Dtype>
void holtwinters_forecast_cpu(Dtype *forecast, int h, int batch_size,
                              int frequency, const Dtype *level_coef,
                              const Dtype *trend_coef, const Dtype *season_coef,
                              aion::SeasonalType seasonal) {
  for (int j = 0; j < batch_size; ++j) {
    const Dtype level = (level_coef) ? level_coef[j] : 0.;
    const Dtype trend = (trend_coef) ? trend_coef[j] : 0.;
    for (int i = 0; i < h; ++i) {
      const Dtype season =
        (season_coef) ? season_coef[j + (i % frequency) * batch_size]
                      : (seasonal == aion::SeasonalType::ADDITIVE) ? 0. : 1.;
      if (seasonal == aion::SeasonalType::ADDITIVE)
        forecast[j + i * batch_size] = level + trend * (i + 1) + season;
      else
        forecast[j + i * batch_size] = (level + trend * (i + 1)) * season;
    }
  }
}

template void holtwinters_forecast_cpu<float>(float *forecast, int h,
                                              int batch_size, int frequency,
                                              const float *level_coef,
                                              const float *trend_coef,
                                              const float *season_coef,
                                              aion::SeasonalType seasonal);
template void holtwinters_forecast_cpu<double>(double *forecast, int h,
                                               int batch_size, int frequency,
                                               const double *level_coef,
                                               const double *trend_coef,
                                               const double *season_coef,
                                               aion::SeasonalType seasonal);

template <typename Dtype>
aion::OptimCriterion holtwinters_bfgs_optim_host(
  int id, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, const Dtype *start_season, const Dtype *beta,
  const Dtype *gamma, bool optim_alpha, bool optim_beta, bool optim_gamma,
  std::vector<Dtype> &x, aion::SeasonalType seasonal,
  const aion::OptimParams<Dtype> optim_params);

template <typename Dtype>
void parabolic_interpolation_golden_optim(
  int id, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, const Dtype *start_season, const Dtype *beta,
  const Dtype *gamma, bool optim_alpha, bool optim_beta, bool optim_gamma,
  std::vector<Dtype> &x, aion::SeasonalType seasonal, Dtype e);

template <typename Dtype>
void holtwinters_optim_cpu(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta, Dtype *gamma,
  bool optim_gamma, Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
  Dtype *error, aion::OptimCriterion *optim_result, aion::SeasonalType seasonal,
  const aion::OptimParams<Dtype> optim_params) {
  for (int j = 0; j < batch_size; ++j) {
    // TODO(ahmad): group init with fit
    int shift = 1;
    std::vector<Dtype> x(3);
    aion::OptimCriterion optim;
    Dtype plevel = start_level[j], ptrend = .0;
    x[0] = alpha[j];
    x[1] = beta ? beta[j] : .0;
    x[2] = gamma ? gamma[j] : .0;

    if (gamma) {
      shift = frequency;
      ptrend = beta ? start_trend[j] : .0;
    } else if (beta) {
      shift = 2;
      ptrend = start_trend[j];
    }

    const int dim = optim_alpha + optim_beta + optim_gamma;
    if (dim == 1)
      parabolic_interpolation_golden_optim<Dtype>(
        j, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
        beta, gamma, optim_alpha, optim_beta, optim_gamma, x, seasonal,
        optim_params.eps);
    else
      optim = holtwinters_bfgs_optim_host<Dtype>(
        j, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
        beta, gamma, optim_alpha, optim_beta, optim_gamma, x, seasonal,
        optim_params);

    if (optim_alpha) alpha[j] = bound_host(x[0]);
    if (optim_beta) beta[j] = bound_host(x[1]);
    if (optim_gamma) gamma[j] = bound_host(x[2]);
    if (dim > 1 && optim_result) optim_result[j] = optim;

    if (error || level || trend || season || xhat) {
      // Final fit
      Dtype error_ = holtwinters_eval_host<Dtype>(
        j, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
        beta, gamma, x[0], x[1], x[2], level, trend, season, xhat, seasonal);
      if (error) error[j] = error_;
    }
  }
}

template <typename Dtype>
Dtype golden_step(Dtype a, Dtype b, Dtype c) {
  Dtype mid = (a + c) * 0.5;
  if (b > mid)
    return (a - b) * GOLD;
  else
    return (c - b) * GOLD;
}

template <typename Dtype>
Dtype fix_step(Dtype a, Dtype b, Dtype c, Dtype step, Dtype e) {
  Dtype min_step = fabs(e * b) + PG_EPS;
  if (fabs(step) < min_step) return step > 0 ? min_step : -min_step;
  if (fabs(b + step - a) <= e || fabs(b + step - c) <= e)
    return 0.0;  // steps are too close to each others
  return step;
}

template <typename Dtype>
Dtype calculate_step(Dtype a, Dtype b, Dtype c, Dtype loss_a, Dtype loss_b,
                     Dtype loss_c, Dtype pstep, Dtype e) {
  // parabola step
  Dtype p = (b - a) * (loss_b - loss_c);
  Dtype q = (b - c) * (loss_b - loss_a);
  Dtype x = q * (b - c) - p * (b - a);
  Dtype y = (p - q) * 2.;
  Dtype step = fabs(y) < PG_EPS ? golden_step(a, b, c) : x / y;
  step = fix_step(a, b, c, step, e);  // ensure point is new

  if (fabs(step) > fabs(pstep / 2) || step == 0.0) step = golden_step(a, b, c);
  return step;
}

template <typename Dtype>
void parabolic_interpolation_golden_optim(
  int id, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, const Dtype *start_season, const Dtype *beta,
  const Dtype *gamma, bool optim_alpha, bool optim_beta, bool optim_gamma,
  std::vector<Dtype> &x, aion::SeasonalType seasonal, Dtype eps) {
  Dtype a = (Dtype).0;
  Dtype b = (Dtype)GOLD;
  Dtype c = (Dtype)1.;

  Dtype loss_a = holtwinters_eval_host<Dtype>(
    id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season, beta,
    gamma, optim_alpha ? a : x[0], optim_beta ? a : x[1],
    optim_gamma ? a : x[2], nullptr, nullptr, nullptr, nullptr, seasonal);
  Dtype loss_b = holtwinters_eval_host<Dtype>(
    id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season, beta,
    gamma, optim_alpha ? b : x[0], optim_beta ? b : x[1],
    optim_gamma ? b : x[2], nullptr, nullptr, nullptr, nullptr, seasonal);
  Dtype loss_c = holtwinters_eval_host<Dtype>(
    id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season, beta,
    gamma, optim_alpha ? c : x[0], optim_beta ? c : x[1],
    optim_gamma ? c : x[2], nullptr, nullptr, nullptr, nullptr, seasonal);

  Dtype pstep = (c - a) / 2;
  Dtype cstep = pstep;

  while (fabs(c - a) > fabs(b * eps) + PG_EPS) {
    Dtype step = calculate_step(a, b, c, loss_a, loss_b, loss_c, cstep, eps);
    Dtype optim_val = b + step;
    Dtype loss_val = holtwinters_eval_host<Dtype>(
      id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
      beta, gamma, optim_alpha ? optim_val : x[0],
      optim_beta ? optim_val : x[1], optim_gamma ? optim_val : x[2], nullptr,
      nullptr, nullptr, nullptr, seasonal);
    if (loss_val < loss_b) {
      if (optim_val > b) {
        SUBSTITUTE(a, b, loss_a, loss_b);
      } else {
        SUBSTITUTE(c, b, loss_c, loss_b);
      }
      SUBSTITUTE(b, optim_val, loss_b, loss_val);
    } else {
      if (optim_val > b) {
        SUBSTITUTE(c, optim_val, loss_c, loss_val);
      } else {
        SUBSTITUTE(a, optim_val, loss_a, loss_val);
      }
    }
    SUBSTITUTE(cstep, pstep, pstep, step);
  }
  if (optim_alpha) x[0] = b;
  if (optim_beta) x[1] = b;
  if (optim_gamma) x[2] = b;
}

template <typename Dtype>
void holtwinters_finite_gradient_host(
  int id, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, const Dtype *start_season, const Dtype *beta,
  const Dtype *gamma, Dtype alpha_, Dtype beta_, Dtype gamma_, Dtype *g_alpha,
  Dtype *g_beta, Dtype *g_gamma, aion::SeasonalType seasonal, Dtype eps) {
  Dtype left_error, right_error;
  if (g_alpha) {  // alpha gradient
    left_error = holtwinters_eval_host<Dtype>(
      id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
      beta, gamma, alpha_ - eps, beta_, gamma_, nullptr, nullptr, nullptr,
      nullptr, seasonal);
    right_error = holtwinters_eval_host<Dtype>(
      id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
      beta, gamma, alpha_ + eps, beta_, gamma_, nullptr, nullptr, nullptr,
      nullptr, seasonal);
    *g_alpha = (right_error - left_error) / (eps * 2.);
  }
  if (g_beta) {  // beta gradient
    left_error = holtwinters_eval_host<Dtype>(
      id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
      beta, gamma, alpha_, beta_ - eps, gamma_, nullptr, nullptr, nullptr,
      nullptr, seasonal);
    right_error = holtwinters_eval_host<Dtype>(
      id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
      beta, gamma, alpha_, beta_ + eps, gamma_, nullptr, nullptr, nullptr,
      nullptr, seasonal);
    *g_beta = (right_error - left_error) / (eps * 2.);
  }
  if (g_gamma) {  // gamma gradient
    left_error = holtwinters_eval_host<Dtype>(
      id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
      beta, gamma, alpha_, beta_, gamma_ - eps, nullptr, nullptr, nullptr,
      nullptr, seasonal);
    right_error = holtwinters_eval_host<Dtype>(
      id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
      beta, gamma, alpha_, beta_, gamma_ + eps, nullptr, nullptr, nullptr,
      nullptr, seasonal);
    *g_gamma = (right_error - left_error) / (eps * 2.);
  }
}

template <typename Dtype>
Dtype linesearch(int dim, int id, const Dtype *ts, int n, int batch_size,
                 int frequency, int shift, Dtype plevel, Dtype ptrend,
                 const Dtype *start_season, const Dtype *beta,
                 const Dtype *gamma, const std::vector<Dtype> &p,
                 const std::vector<Dtype> &g, const std::vector<Dtype> &x,
                 std::vector<Dtype> &nx, Dtype *loss_diff,
                 const aion::OptimParams<Dtype> optim_params,
                 aion::SeasonalType seasonal) {
  // {next_params} = {params}+step_size*p;
  Dtype step_size = optim_params.linesearch_step_size;
  if (step_size <= 0)
    step_size =
      (Dtype)0.866 / sqrt(aion::cblas::dot(dim, p.data(), 1, p.data(), 1));
  nx = x;
  aion::cblas::axpy<Dtype>(dim, step_size, p.data(), nx.data());

  const Dtype cauchy = optim_params.linesearch_c *
                       aion::cblas::dot<Dtype>(dim, g.data(), 1, p.data(), 1);
  const Dtype loss_ref = holtwinters_eval_host<Dtype>(
    id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season, beta,
    gamma, x[0], x[1], x[2], nullptr, nullptr, nullptr, nullptr, seasonal);
  Dtype loss = holtwinters_eval_host<Dtype>(
    id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season, beta,
    gamma, nx[0], nx[1], nx[2], nullptr, nullptr, nullptr, nullptr, seasonal);

  for (int i = 0; (i < optim_params.linesearch_iter_limit) &&
                  (loss > loss_ref + step_size * cauchy);
       ++i) {
    step_size *= optim_params.linesearch_tau;
    nx = x;
    aion::cblas::axpy<Dtype>(dim, step_size, p.data(), nx.data());
    loss = holtwinters_eval_host<Dtype>(id, ts, n, batch_size, frequency, shift,
                                        plevel, ptrend, start_season, beta,
                                        gamma, nx[0], nx[1], nx[2], nullptr,
                                        nullptr, nullptr, nullptr, seasonal);
  }

  *loss_diff = std::abs(loss - loss_ref);
  return step_size;
}

// TODO(ahmad): use ?syr instead (H is symmetric)
template <typename Dtype>
aion::OptimCriterion holtwinters_bfgs_optim_host(
  int id, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, const Dtype *start_season, const Dtype *beta,
  const Dtype *gamma, bool optim_alpha, bool optim_beta, bool optim_gamma,
  std::vector<Dtype> &x, aion::SeasonalType seasonal,
  const aion::OptimParams<Dtype> optim_params) {
  const int dim = 3;
  std::vector<Dtype> nx(dim, .0);       // next params
  std::vector<Dtype> g(dim, .0);        // gradients
  std::vector<Dtype> ng(dim, .0);       // next gradient
  std::vector<Dtype> H(dim * dim, .0);  // Hessian
  std::vector<Dtype> p(dim, .0);        // direction
  std::vector<Dtype> s(dim, .0);
  std::vector<Dtype> y(dim, .0);
  std::vector<Dtype> tmp_buffer(dim, .0);

  for (int i = 0; i < dim; ++i) H[i + i * dim] = 1.;

  holtwinters_finite_gradient_host<Dtype>(
    id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season, beta,
    gamma, x[0], x[1], x[2], optim_alpha ? &g[0] : nullptr,
    optim_beta ? &g[1] : nullptr, optim_gamma ? &g[2] : nullptr, seasonal,
    optim_params.eps);

  for (int iter = 0; iter < optim_params.bfgs_iter_limit; ++iter) {
    // p = -H*gradient;
    aion::cblas::gemv(CblasNoTrans, dim, dim, (Dtype)-1., H.data(), dim,
                      g.data(), 1, (Dtype)0., p.data(), 1);

    Dtype phi = aion::cblas::dot<Dtype>(dim, g.data(), 1, p.data(), 1);
    if (phi > 0) {
      // H=Iden(3x3)
      std::fill(H.begin(), H.end(), .0);
      for (int i = 0; i < dim; ++i) H[i + i * dim] = 1.;
      // p = -gradient;
      std::fill(p.begin(), p.end(), .0);
      aion::cblas::axpy<Dtype>(dim, (Dtype)-1, g.data(), p.data());
    }

    // {next_params} = {params}+step_size*p;
    Dtype loss_diff;
    Dtype step_size = linesearch(dim, id, ts, n, batch_size, frequency, shift,
                                 plevel, ptrend, start_season, beta, gamma, p,
                                 g, x, nx, &loss_diff, optim_params, seasonal);

    // see if new {prams} meet stop condition
    Dtype max = -1;
    for (int i = 0; i < dim; ++i) {
      Dtype abs_diff = std::abs(nx[i] - x[i]);
      max = abs_diff > max ? abs_diff : max;
    }
    x = nx;  // update {params}
    if (optim_params.min_param_diff > max)
      return aion::OptimCriterion::OPTIM_MIN_PARAM_DIFF;
    if (optim_params.min_error_diff < loss_diff)
      return aion::OptimCriterion::OPTIM_MIN_ERROR_DIFF;

    holtwinters_finite_gradient_host<Dtype>(
      id, ts, n, batch_size, frequency, shift, plevel, ptrend, start_season,
      beta, gamma, nx[0], nx[1], nx[2], optim_alpha ? &ng[0] : nullptr,
      optim_beta ? &ng[1] : nullptr, optim_gamma ? &ng[2] : nullptr, seasonal,
      optim_params.eps);
    // see if new gradients meet stop condition
    max = -1;
    for (int i = 0; i < dim; ++i) {
      Dtype abs_ng = fabs(ng[i]);
      max = abs_ng > max ? abs_ng : max;
    }
    if (max < optim_params.min_grad_norm)
      return aion::OptimCriterion::OPTIM_MIN_GRAD_NORM;

    // s = step_size*p;
    std::fill(s.begin(), s.end(), .0);
    aion::cblas::axpy<Dtype>(dim, step_size, p.data(), s.data());

    // y = next_grad-grad
    y = ng;
    aion::cblas::axpy<Dtype>(dim, (Dtype)-1., g.data(), y.data());

    const Dtype rho_ = aion::cblas::dot<Dtype>(dim, y.data(), 1, s.data(), 1);
    const Dtype rho = 1.0 / rho_;

    aion::cblas::gemv<Dtype>(CblasNoTrans, dim, dim, (Dtype)1., H.data(), dim,
                             y.data(), 1, (Dtype)0., tmp_buffer.data(),
                             1);  // Hy = H*y
    const Dtype yHy =
      aion::cblas::dot<Dtype>(dim, y.data(), 1, tmp_buffer.data(), 1);

    // H = H + [(rho+zeta)/rho^2]*[s*st]
    aion::cblas::ger<Dtype>(dim, dim, (rho_ + yHy) * (rho * rho), s.data(), 1,
                            s.data(), 1, H.data(), dim);

    // TODO(ahmad): use SSYR2 instead (H is symmetric)
    // H = H + [(rho+zeta)/rho^2]*[Hy*st+s*yt*H]
    aion::cblas::ger<Dtype>(dim, dim, -rho, tmp_buffer.data(), 1, s.data(), 1,
                            H.data(), dim);
    aion::cblas::ger<Dtype>(dim, dim, -rho, s.data(), 1, tmp_buffer.data(), 1,
                            H.data(), dim);

    g = ng;
  }

  return aion::OPTIM_BFGS_ITER_LIMIT;
}

template void holtwinters_optim_cpu<float>(
  const float *ts, int n, int batch_size, int frequency,
  const float *start_level, const float *start_trend, const float *start_season,
  float *alpha, bool optim_alpha, float *beta, bool optim_beta, float *gamma,
  bool optim_gamma, float *level, float *trend, float *season, float *xhat,
  float *error, aion::OptimCriterion *optim_result, aion::SeasonalType seasonal,
  const aion::OptimParams<float> optim_params);
template void holtwinters_optim_cpu<double>(
  const double *ts, int n, int batch_size, int frequency,
  const double *start_level, const double *start_trend,
  const double *start_season, double *alpha, bool optim_alpha, double *beta,
  bool optim_beta, double *gamma, bool optim_gamma, double *level,
  double *trend, double *season, double *xhat, double *error,
  aion::OptimCriterion *optim_result, aion::SeasonalType seasonal,
  const aion::OptimParams<double> optim_params);

template void holtwinters_finite_gradient_host<float>(
  int id, const float *ts, int n, int batch_size, int frequency, int shift,
  float plevel, float ptrend, const float *start_season, const float *beta,
  const float *gamma, float alpha_, float beta_, float gamma_, float *g_alpha,
  float *g_beta, float *g_gamma, aion::SeasonalType seasonal, float eps);
template void holtwinters_finite_gradient_host<double>(
  int id, const double *ts, int n, int batch_size, int frequency, int shift,
  double plevel, double ptrend, const double *start_season, const double *beta,
  const double *gamma, double alpha_, double beta_, double gamma_,
  double *g_alpha, double *g_beta, double *g_gamma, aion::SeasonalType seasonal,
  double eps);
