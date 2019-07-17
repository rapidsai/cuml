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

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "HoltWinters.hpp"
#include "holtwinters_utils.hpp"
#include "hw_cu_utils.hpp"
#include "hw_math.hpp"
#include "utils.h"

#define IDX(n, m, N) (n + (m) * (N))

// TODO(ahmad): n is unused
template <typename Dtype>
void stl_decomposition_gpu(const Dtype *ts, int n, int batch_size,
                           int frequency, int start_periods, Dtype *start_level,
                           Dtype *start_trend, Dtype *start_season,
                           ML::SeasonalType seasonal) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int end = start_periods * frequency;
  const int filter_size = (frequency / 2) * 2 + 1;
  const int trend_len = end - filter_size + 1;

  // Set filter
  std::vector<Dtype> filter_h(filter_size, 1. / frequency);
  if (frequency % 2 == 0) {
    filter_h.front() /= 2;
    filter_h.back() /= 2;
  }
  Dtype *filter_d;
  MLCommon::allocate(filter_d, filter_size, stream);
  MLCommon::updateDevice(filter_d, filter_h.data(), filter_size, stream);

  // Set Trend
  Dtype *trend_d;
  MLCommon::allocate(trend_d, batch_size * trend_len, stream);
  conv1d<Dtype>(ts, batch_size, filter_d, filter_size, trend_d, trend_len);

  Dtype *season_d;
  MLCommon::allocate(season_d, batch_size * trend_len, stream);

  const int ts_offset = (filter_size / 2) * batch_size;
  if (seasonal == ML::SeasonalType::ADDITIVE) {
    const Dtype one = 1.;
    const Dtype minus_one = -1.;
    ML::cublas::geam<Dtype>(CUBLAS_OP_N, CUBLAS_OP_N, trend_len, batch_size,
                            &one, ts + ts_offset, trend_len, &minus_one,
                            trend_d, trend_len, season_d, trend_len);
  } else {
    ML::HoltWinters::Math::div_gpu<Dtype>(trend_len * batch_size,
                                          ts + ts_offset, trend_d, season_d);
  }

  season_mean(season_d, trend_len, batch_size, start_season, frequency,
              filter_size / 2, seasonal);  // TODO(ahmad): return

  batched_ls(trend_d, trend_len, batch_size, start_level,
             start_trend);  // TODO(ahmad): return

  CUDA_CHECK(cudaFree(filter_d));
  CUDA_CHECK(cudaFree(trend_d));
  CUDA_CHECK(cudaFree(season_d));
}

// TODO(ahmad): optimize, maybe im2col ?
template <typename Dtype>
__global__ void conv1d_kernel(const Dtype *input, int batch_size,
                              const Dtype *filter, int filter_size,
                              Dtype *output, int output_size) {
  const int tid = GET_TID;
  if (tid < batch_size) {
    for (int o = 0; o < output_size; ++o) {
      Dtype out = 0.;
      for (int i = 0; i < filter_size; ++i)
        out += filter[i] * input[tid + (i + o) * batch_size];
      output[tid + o * batch_size] = out;
    }
  }
}

template <typename Dtype>
void conv1d(const Dtype *input, int batch_size, const Dtype *filter,
            int filter_size, Dtype *output, int output_size) {
  int total_threads = batch_size;
  conv1d_kernel<Dtype>
    <<<GET_NUM_BLOCKS(total_threads), GET_THREADS_PER_BLOCK(total_threads)>>>(
      input, batch_size, filter, filter_size, output, output_size);
}

// TODO(ahmad): optimize
template <typename Dtype, bool ADDITIVE_KERNEL>
__global__ void season_mean_kernel(const Dtype *season, int len, int batch_size,
                                   Dtype *start_season, int frequency,
                                   int half_filter_size) {
  int tid = GET_TID;
  if (tid < batch_size) {
    Dtype mean = 0.0;
    for (int i = 0; i < frequency; ++i) {
      Dtype period_mean = 0.;
      int c = 0;
      for (int k = i; k < len; k = k + frequency) {
        period_mean += season[k * batch_size + tid];
        c++;
      }
      int count = 1 + ((len - i - 1) / frequency);
      period_mean /= count;
      int ss_idx = (i + half_filter_size) % frequency;
      start_season[ss_idx * batch_size + tid] = period_mean;
      mean += period_mean;
    }
    mean /= frequency;
    for (int i = 0; i < frequency; ++i) {
      if (ADDITIVE_KERNEL)
        start_season[i * batch_size + tid] -= mean;
      else  // MULTIPLOCATIVE
        start_season[i * batch_size + tid] /= mean;
    }
  }
}

template <typename Dtype>
void season_mean(const Dtype *season, int len, int batch_size,
                 Dtype *start_season, int frequency, int half_filter_size,
                 ML::SeasonalType seasonal) {
  if (seasonal == ML::SeasonalType::ADDITIVE)
    season_mean_kernel<Dtype, true>
      <<<GET_NUM_BLOCKS(batch_size), GET_THREADS_PER_BLOCK(batch_size)>>>(
        season, len, batch_size, start_season, frequency, half_filter_size);
  else
    season_mean_kernel<Dtype, false>
      <<<GET_NUM_BLOCKS(batch_size), GET_THREADS_PER_BLOCK(batch_size)>>>(
        season, len, batch_size, start_season, frequency, half_filter_size);
}

template <typename Dtype>
__global__ void RinvKernel(const Dtype *A, Dtype *Rinv, int trend_len) {
  // Inverse of R (2x2 upper triangular matrix)
  int tid = GET_TID;
  if (tid == 0) {
    Dtype a = A[0], b = A[trend_len], d = A[trend_len + 1];
    Dtype factor = 1. / (a * d);
    Rinv[0] = factor * d;
    Rinv[1] = 0.;
    Rinv[2] = -factor * b;
    Rinv[3] = factor * a;
  }
}

template <typename Dtype>
__global__ void batched_ls_solver_kernel(const Dtype *B, const Dtype *rq,
                                         int batch_size, int len, Dtype *level,
                                         Dtype *trend) {
  int tid = GET_TID;
  if (tid < batch_size) {
    Dtype level_ = 0., trend_ = 0.;
    for (int i = 0; i < len; ++i) {
      Dtype b = B[tid + i * batch_size];
      level_ += rq[2 * i] * b;
      trend_ += rq[2 * i + 1] * b;
    }
    level[tid] = level_;
    trend[tid] = trend_;
  }
}

template <typename Dtype>
void batched_ls(const Dtype *data, int trend_len, int batch_size, Dtype *level,
                Dtype *trend) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const Dtype one = (Dtype)1.;
  const Dtype zero = (Dtype)0.;
  int geqrf_buffer;
  int orgqr_buffer;
  int lwork_size;

  // Allocate memory
  std::vector<Dtype> A_h(2 * trend_len);
  Dtype *A_d = nullptr, *tau_d = nullptr, *Rinv_d = nullptr, *R1Qt_d = nullptr,
        *lwork_d = nullptr;
  int *dev_info_d = nullptr;

  MLCommon::allocate(A_d, 2 * trend_len, stream);
  MLCommon::allocate(tau_d, 2, stream);
  MLCommon::allocate(Rinv_d, 4, stream);
  MLCommon::allocate(R1Qt_d, 2 * trend_len, stream);
  MLCommon::allocate(dev_info_d, 1, stream);

  // Prepare A
  for (int i = 0; i < trend_len; ++i) {
    A_h[i] = (Dtype)1.;
    A_h[trend_len + i] = (Dtype)(i + 1);
  }
  MLCommon::updateDevice(A_d, A_h.data, 2 * trend_len, stream);

  ML::cusolver::geqrf_bufferSize<Dtype>(trend_len, 2, A_d, 2, &geqrf_buffer);
  ML::cusolver::orgqr_bufferSize<Dtype>(trend_len, 2, 2, A_d, 2, tau_d,
                                        &orgqr_buffer);
  lwork_size = geqrf_buffer > orgqr_buffer ? geqrf_buffer : orgqr_buffer;
  MLCommon::allocate(lwork_d, lwork_size, stream);

  // QR decomposition of A
  // TODO(ahmad): return value
  ML::cusolver::geqrf<Dtype>(trend_len, 2, A_d, trend_len, tau_d, lwork_d,
                             lwork_size, dev_info_d);

  // Single thread kenrel to inverse R
  RinvKernel<Dtype><<<1, 1>>>(A_d, Rinv_d, trend_len);

  // R1QT = inv(R)*transpose(Q)
  // TODO(ahmad): return value
  ML::cusolver::orgqr<Dtype>(trend_len, 2, 2, A_d, trend_len, tau_d, lwork_d,
                             lwork_size, dev_info_d);
  ML::cublas::gemm<Dtype>(CUBLAS_OP_N, CUBLAS_OP_T, 2, trend_len, 2, &one,
                          Rinv_d, 2, A_d, trend_len, &zero, R1Qt_d,
                          2);  // TODO(ahmad): return value

  batched_ls_solver_kernel<Dtype>
    <<<GET_NUM_BLOCKS(batch_size), GET_THREADS_PER_BLOCK(batch_size)>>>(
      data, R1Qt_d, batch_size, trend_len, level, trend);
}

template <typename Dtype, bool additive_seasonal>
__global__ void holtwinters_eval_gpu_global_kernel(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *pseason, const Dtype *alpha, const Dtype *beta, const Dtype *gamma,
  Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat, Dtype *error);
template <typename Dtype, bool additive_seasonal>
__global__ void holtwinters_eval_gpu_shared_kernel(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  const Dtype *alpha, const Dtype *beta, const Dtype *gamma, Dtype *level,
  Dtype *trend, Dtype *season, Dtype *xhat, Dtype *error);

template <>
__device__ float bound_device<float>(float val, float min, float max) {
  return fminf(fmaxf(val, min), max);
}
template <>
__device__ double bound_device<double>(double val, double min, double max) {
  return fmin(fmax(val, min), max);
}

template <typename Dtype>
void holtwinters_eval_gpu(const Dtype *ts, int n, int batch_size, int frequency,
                          const Dtype *start_level, const Dtype *start_trend,
                          const Dtype *start_season, const Dtype *alpha,
                          const Dtype *beta, const Dtype *gamma, Dtype *level,
                          Dtype *trend, Dtype *season, Dtype *xhat,
                          Dtype *error, ML::SeasonalType seasonal) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int total_blocks = GET_NUM_BLOCKS(batch_size);
  int threads_per_block = GET_THREADS_PER_BLOCK(batch_size);

  // Get shared memory size
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  struct cudaDeviceProp prop;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // How much sm needed for shared kernel
  size_t sm_needed = sizeof(Dtype) * threads_per_block * frequency;

  if (sm_needed >
      prop.sharedMemPerBlock) {  // TODO(ahmad): test shared/general kernels
    Dtype *pseason;
    MLCommon::allocate(pseason, batch_size * frequency);
    if (seasonal == ML::SeasonalType::ADDITIVE)
      holtwinters_eval_gpu_global_kernel<Dtype, true>
        <<<total_blocks, threads_per_block>>>(
          ts, n, batch_size, frequency, start_level, start_trend, start_season,
          pseason, alpha, beta, gamma, level, trend, season, xhat, error);
    else
      holtwinters_eval_gpu_global_kernel<Dtype, false>
        <<<total_blocks, threads_per_block>>>(
          ts, n, batch_size, frequency, start_level, start_trend, start_season,
          pseason, alpha, beta, gamma, level, trend, season, xhat, error);
    CUDA_CHECK(cudaFree(pseason));
  } else {
    if (seasonal == ML::SeasonalType::ADDITIVE)
      holtwinters_eval_gpu_shared_kernel<Dtype, true>
        <<<total_blocks, threads_per_block, sm_needed>>>(
          ts, n, batch_size, frequency, start_level, start_trend, start_season,
          alpha, beta, gamma, level, trend, season, xhat, error);
    else
      holtwinters_eval_gpu_shared_kernel<Dtype, false>
        <<<total_blocks, threads_per_block, sm_needed>>>(
          ts, n, batch_size, frequency, start_level, start_trend, start_season,
          alpha, beta, gamma, level, trend, season, xhat, error);
  }
}

template <typename Dtype, bool additive_seasonal>
__device__ Dtype holtwinters_eval_device(
  int tid, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, Dtype *pseason, int pseason_width,
  const Dtype *start_season, const Dtype *beta, const Dtype *gamma,
  Dtype alpha_, Dtype beta_, Dtype gamma_, Dtype *level, Dtype *trend,
  Dtype *season, Dtype *xhat) {
  alpha_ = bound_device(alpha_);
  beta_ = bound_device(beta_);
  gamma_ = bound_device(gamma_);

  Dtype error_ = .0;
  Dtype clevel = .0, ctrend = .0, cseason = .0;
  for (int i = 0; i < n - shift; i++) {
    int s = i % frequency;
    Dtype pts = ts[IDX(tid, i + shift, batch_size)];
    Dtype leveltrend = plevel + ptrend;

    // xhat
    Dtype stmp;
    if (gamma)
      stmp = i < frequency ? start_season[IDX(tid, i, batch_size)]
                           : pseason[s * pseason_width];
    else
      stmp = (!additive_seasonal);
    Dtype xhat_ = plevel + ptrend;
    if (additive_seasonal)
      xhat_ += stmp;
    else
      xhat_ *= stmp;

    // Error
    Dtype diff = pts - xhat_;
    error_ += diff * diff;

    // Level
    if (additive_seasonal) {
      clevel = alpha_ * (pts - stmp) + (1 - alpha_) * (leveltrend);
    } else {
      Dtype stmp_eps = abs(stmp) > STMP_EPS ? stmp : STMP_EPS;
      clevel = alpha_ * (pts / stmp_eps) + (1 - alpha_) * (leveltrend);
    }

    // Trend
    if (beta) {
      ctrend = beta_ * (clevel - plevel) + (1 - beta_) * ptrend;
      ptrend = ctrend;
    }

    // Seasonal
    if (gamma) {
      if (additive_seasonal)
        cseason = gamma_ * (pts - clevel) + (1 - gamma_) * stmp;
      else
        cseason = gamma_ * (pts / clevel) + (1 - gamma_) * stmp;
      pseason[s * pseason_width] = cseason;
    }

    plevel = clevel;

    if (level) level[IDX(tid, i, batch_size)] = clevel;
    if (trend) trend[IDX(tid, i, batch_size)] = ctrend;
    if (season) season[IDX(tid, i, batch_size)] = cseason;
    if (xhat) xhat[IDX(tid, i, batch_size)] = xhat_;
  }
  return error_;
}

template <typename Dtype, bool additive_seasonal>
__global__ void holtwinters_eval_gpu_shared_kernel(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  const Dtype *alpha, const Dtype *beta, const Dtype *gamma, Dtype *level,
  Dtype *trend, Dtype *season, Dtype *xhat, Dtype *error) {
  int tid = GET_TID;
  extern __shared__ __align__(sizeof(Dtype)) unsigned char pseason_[];
  Dtype *pseason = reinterpret_cast<Dtype *>(pseason_);

  if (tid < batch_size) {
    int shift = 1;
    Dtype plevel = start_level[tid], ptrend = .0;
    Dtype alpha_ = alpha[tid];
    Dtype beta_ = beta ? beta[tid] : .0;
    Dtype gamma_ = gamma ? gamma[tid] : .0;

    if (gamma) {
      shift = frequency;
      ptrend = beta ? start_trend[tid] : .0;
    } else if (beta) {
      shift = 2;
      ptrend = start_trend[tid];
    }

    Dtype error_ = holtwinters_eval_device<Dtype, additive_seasonal>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend,
      pseason + threadIdx.x, blockDim.x, start_season, beta, gamma, alpha_,
      beta_, gamma_, level, trend, season, xhat);
    if (error) error[tid] = error_;
  }
}

template <typename Dtype, bool additive_seasonal>
__global__ void holtwinters_eval_gpu_global_kernel(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *pseason, const Dtype *alpha, const Dtype *beta, const Dtype *gamma,
  Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat, Dtype *error) {
  int tid = GET_TID;

  if (tid < batch_size) {
    int shift = 1;
    Dtype plevel = start_level[tid], ptrend = .0;
    Dtype alpha_ = alpha[tid];
    Dtype beta_ = beta ? beta[tid] : .0;
    Dtype gamma_ = gamma ? gamma[tid] : .0;

    if (gamma) {
      shift = frequency;
      ptrend = beta ? start_trend[tid] : .0;
    } else if (beta) {
      shift = 2;
      ptrend = start_trend[tid];
    }

    Dtype error_ = holtwinters_eval_device<Dtype, additive_seasonal>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason + tid,
      batch_size, start_season, beta, gamma, alpha_, beta_, gamma_, level,
      trend, season, xhat);
    if (error) error[tid] = error_;
  }
}

template <typename Dtype, bool additive>
__global__ void holtwinters_seasonal_forecast_kernel(
  Dtype *forecast, int h, int batch_size, int frequency,
  const Dtype *level_coef, const Dtype *trend_coef, const Dtype *season_coef) {
  int tid = GET_TID;
  if (tid < batch_size) {
    const Dtype level = (level_coef) ? level_coef[tid] : 0.;
    const Dtype trend = (trend_coef) ? trend_coef[tid] : 0.;
    for (int i = 0; i < h; ++i) {
      const Dtype season = season_coef[tid + (i % frequency) * batch_size];
      if (additive)
        forecast[tid + i * batch_size] = level + trend * (i + 1) + season;
      else
        forecast[tid + i * batch_size] = (level + trend * (i + 1)) * season;
    }
  }
}

template <typename Dtype>
__global__ void holtwinters_nonseasonal_forecast_kernel(
  Dtype *forecast, int h, int batch_size, const Dtype *level_coef,
  const Dtype *trend_coef) {
  int tid = GET_TID;
  if (tid < batch_size) {
    const Dtype level = (level_coef) ? level_coef[tid] : 0.;
    const Dtype trend = trend_coef[tid];
    for (int i = 0; i < h; ++i)
      forecast[tid + i * batch_size] = level + trend * (i + 1);
  }
}

template <typename Dtype>
__global__ void holtwinters_level_forecast_kernel(Dtype *forecast, int h,
                                                  int batch_size,
                                                  const Dtype *level_coef) {
  int tid = GET_TID;
  if (tid < batch_size) {
    const Dtype level = level_coef[tid];
    for (int i = 0; i < h; ++i) forecast[tid + i * batch_size] = level;
  }
}

template <typename Dtype>
void holtwinters_forecast_gpu(Dtype *forecast, int h, int batch_size,
                              int frequency, const Dtype *level_coef,
                              const Dtype *trend_coef, const Dtype *season_coef,
                              ML::SeasonalType seasonal) {
  int total_blocks = GET_NUM_BLOCKS(batch_size);
  int threads_per_block = GET_THREADS_PER_BLOCK(batch_size);

  if (trend_coef == nullptr && season_coef == nullptr) {
    holtwinters_level_forecast_kernel<Dtype>
      <<<total_blocks, threads_per_block>>>(forecast, h, batch_size,
                                            level_coef);
  } else if (season_coef == nullptr) {
    holtwinters_nonseasonal_forecast_kernel<Dtype>
      <<<total_blocks, threads_per_block>>>(forecast, h, batch_size, level_coef,
                                            trend_coef);
  } else {
    if (seasonal == ML::SeasonalType::ADDITIVE)
      holtwinters_seasonal_forecast_kernel<Dtype, true>
        <<<total_blocks, threads_per_block>>>(forecast, h, batch_size,
                                              frequency, level_coef, trend_coef,
                                              season_coef);
    else
      holtwinters_seasonal_forecast_kernel<Dtype, false>
        <<<total_blocks, threads_per_block>>>(forecast, h, batch_size,
                                              frequency, level_coef, trend_coef,
                                              season_coef);
  }
}

template <typename Dtype, bool ADDITIVE_KERNEL, bool single_param>
__global__ void holtwinters_optim_gpu_shared_kernel(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta, Dtype *gamma,
  bool optim_gamma, Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
  Dtype *error, ML::OptimCriterion *optim_result,
  const ML::OptimParams<Dtype> optim_params);

template <typename Dtype, bool ADDITIVE_KERNEL, bool single_param>
__global__ void holtwinters_optim_gpu_global_kernel(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *pseason, Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta,
  Dtype *gamma, bool optim_gamma, Dtype *level, Dtype *trend, Dtype *season,
  Dtype *xhat, Dtype *error, ML::OptimCriterion *optim_result,
  const ML::OptimParams<Dtype> optim_params);

template <typename Dtype, bool ADDITIVE_KERNEL>
__device__ void parabolic_interpolation_golden_optim(
  int tid, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, Dtype *pseason, int pseason_width,
  const Dtype *start_season, const Dtype *beta, const Dtype *gamma,
  bool optim_alpha, Dtype *alpha_, bool optim_beta, Dtype *beta_,
  bool optim_gamma, Dtype *gamma_, Dtype eps);

template <typename Dtype, bool ADDITIVE_KERNEL>
__device__ ML::OptimCriterion holtwinters_bfgs_optim_device(
  int tid, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, Dtype *pseason, int pseason_width,
  const Dtype *start_season, const Dtype *beta, const Dtype *gamma,
  bool optim_alpha, Dtype *x1, bool optim_beta, Dtype *x2, bool optim_gamma,
  Dtype *x3, const ML::OptimParams<Dtype> optim_params);

template <typename Dtype>
__device__ Dtype max3(Dtype a, Dtype b, Dtype c) {
  return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

template <typename Dtype>
void holtwinters_optim_gpu(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta, Dtype *gamma,
  bool optim_gamma, Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
  Dtype *error, ML::OptimCriterion *optim_result, ML::SeasonalType seasonal,
  const ML::OptimParams<Dtype> optim_params) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  //int total_blocks = GET_NUM_BLOCKS(batch_size);
  //int threads_per_block = GET_THREADS_PER_BLOCK(batch_size);
  int total_blocks = (batch_size - 1) / 128 + 1;
  int threads_per_block = 128;

  // Get shared memory size // TODO(ahmad) put into a function
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  struct cudaDeviceProp prop;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // How much sm needed for shared kernel
  size_t sm_needed = sizeof(Dtype) * threads_per_block * frequency;

  if (
    sm_needed >
    prop
      .sharedMemPerBlock) {  // Global memory // TODO(ahmad): test shared/general kernels
    Dtype *pseason;
    MLCommon::allocate(pseason, batch_size * frequency, stream);
    if (seasonal == ML::SeasonalType::ADDITIVE) {
      if (optim_alpha + optim_beta + optim_gamma > 1)
        holtwinters_optim_gpu_global_kernel<Dtype, true, false>
          <<<total_blocks, threads_per_block>>>(
            ts, n, batch_size, frequency, start_level, start_trend,
            start_season, pseason, alpha, optim_alpha, beta, optim_beta, gamma,
            optim_gamma, level, trend, season, xhat, error, optim_result,
            optim_params);
      else
        holtwinters_optim_gpu_global_kernel<Dtype, true, true>
          <<<total_blocks, threads_per_block>>>(
            ts, n, batch_size, frequency, start_level, start_trend,
            start_season, pseason, alpha, optim_alpha, beta, optim_beta, gamma,
            optim_gamma, level, trend, season, xhat, error, optim_result,
            optim_params);
    } else {
      if (optim_alpha + optim_beta + optim_gamma > 1)
        holtwinters_optim_gpu_global_kernel<Dtype, false, false>
          <<<total_blocks, threads_per_block>>>(
            ts, n, batch_size, frequency, start_level, start_trend,
            start_season, pseason, alpha, optim_alpha, beta, optim_beta, gamma,
            optim_gamma, level, trend, season, xhat, error, optim_result,
            optim_params);
      else
        holtwinters_optim_gpu_global_kernel<Dtype, false, true>
          <<<total_blocks, threads_per_block>>>(
            ts, n, batch_size, frequency, start_level, start_trend,
            start_season, pseason, alpha, optim_alpha, beta, optim_beta, gamma,
            optim_gamma, level, trend, season, xhat, error, optim_result,
            optim_params);
    }
    CUDA_CHECK(cudaFree(pseason));
  } else {  // Shared memory
    if (seasonal == ML::SeasonalType::ADDITIVE) {
      if (optim_alpha + optim_beta + optim_gamma > 1)
        holtwinters_optim_gpu_shared_kernel<Dtype, true, false>
          <<<total_blocks, threads_per_block, sm_needed>>>(
            ts, n, batch_size, frequency, start_level, start_trend,
            start_season, alpha, optim_alpha, beta, optim_beta, gamma,
            optim_gamma, level, trend, season, xhat, error, optim_result,
            optim_params);
      else
        holtwinters_optim_gpu_shared_kernel<Dtype, true, true>
          <<<total_blocks, threads_per_block, sm_needed>>>(
            ts, n, batch_size, frequency, start_level, start_trend,
            start_season, alpha, optim_alpha, beta, optim_beta, gamma,
            optim_gamma, level, trend, season, xhat, error, optim_result,
            optim_params);
    } else {
      if (optim_alpha + optim_beta + optim_gamma > 1)
        holtwinters_optim_gpu_shared_kernel<Dtype, false, false>
          <<<total_blocks, threads_per_block, sm_needed>>>(
            ts, n, batch_size, frequency, start_level, start_trend,
            start_season, alpha, optim_alpha, beta, optim_beta, gamma,
            optim_gamma, level, trend, season, xhat, error, optim_result,
            optim_params);
      else
        holtwinters_optim_gpu_shared_kernel<Dtype, false, true>
          <<<total_blocks, threads_per_block, sm_needed>>>(
            ts, n, batch_size, frequency, start_level, start_trend,
            start_season, alpha, optim_alpha, beta, optim_beta, gamma,
            optim_gamma, level, trend, season, xhat, error, optim_result,
            optim_params);
    }
  }
}

template <typename Dtype, bool ADDITIVE_KERNEL, bool single_param>
__global__ void holtwinters_optim_gpu_shared_kernel(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta, Dtype *gamma,
  bool optim_gamma, Dtype *level, Dtype *trend, Dtype *season, Dtype *xhat,
  Dtype *error, ML::OptimCriterion *optim_result,
  const ML::OptimParams<Dtype> optim_params) {
  int tid = GET_TID;
  extern __shared__ __align__(sizeof(Dtype)) unsigned char pseason_[];
  Dtype *pseason = reinterpret_cast<Dtype *>(pseason_);

  if (tid < batch_size) {
    // TODO(ahmad): group init with fit
    int shift = 1;
    ML::OptimCriterion optim;
    Dtype plevel = start_level[tid], ptrend = .0;
    Dtype alpha_ = alpha[tid];
    Dtype beta_ = beta ? beta[tid] : .0;
    Dtype gamma_ = gamma ? gamma[tid] : .0;

    if (gamma) {
      shift = frequency;
      ptrend = beta ? start_trend[tid] : .0;
    } else if (beta) {
      shift = 2;
      ptrend = start_trend[tid];
    }

    // Optimization
    if (single_param)
      parabolic_interpolation_golden_optim<Dtype, ADDITIVE_KERNEL>(
        tid, ts, n, batch_size, frequency, shift, plevel, ptrend,
        pseason + threadIdx.x, blockDim.x, start_season, beta, gamma,
        optim_alpha, &alpha_, optim_beta, &beta_, optim_gamma, &gamma_,
        optim_params.eps);
    else
      optim = holtwinters_bfgs_optim_device<Dtype, ADDITIVE_KERNEL>(
        tid, ts, n, batch_size, frequency, shift, plevel, ptrend,
        pseason + threadIdx.x, blockDim.x, start_season, beta, gamma,
        optim_alpha, &alpha_, optim_beta, &beta_, optim_gamma, &gamma_,
        optim_params);

    if (optim_alpha) alpha[tid] = bound_device(alpha_);
    if (optim_beta) beta[tid] = bound_device(beta_);
    if (optim_gamma) gamma[tid] = bound_device(gamma_);
    if (!single_param && optim_result) optim_result[tid] = optim;

    if (error || level || trend || season || xhat) {
      // Final fit
      Dtype error_ = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
        tid, ts, n, batch_size, frequency, shift, plevel, ptrend,
        pseason + threadIdx.x, blockDim.x, start_season, beta, gamma, alpha_,
        beta_, gamma_, level, trend, season, xhat);
      if (error) error[tid] = error_;
    }
  }
}

template <typename Dtype, bool ADDITIVE_KERNEL, bool single_param>
__global__ void holtwinters_optim_gpu_global_kernel(
  const Dtype *ts, int n, int batch_size, int frequency,
  const Dtype *start_level, const Dtype *start_trend, const Dtype *start_season,
  Dtype *pseason, Dtype *alpha, bool optim_alpha, Dtype *beta, bool optim_beta,
  Dtype *gamma, bool optim_gamma, Dtype *level, Dtype *trend, Dtype *season,
  Dtype *xhat, Dtype *error, ML::OptimCriterion *optim_result,
  const ML::OptimParams<Dtype> optim_params) {
  int tid = GET_TID;
  if (tid < batch_size) {
    // TODO(ahmad): group init with fit
    int shift = 1;
    ML::OptimCriterion optim;
    Dtype plevel = start_level[tid], ptrend = .0;
    Dtype alpha_ = alpha[tid];
    Dtype beta_ = beta ? beta[tid] : .0;
    Dtype gamma_ = gamma ? gamma[tid] : .0;

    if (gamma) {
      shift = frequency;
      ptrend = beta ? start_trend[tid] : .0;
    } else if (beta) {
      shift = 2;
      ptrend = start_trend[tid];
    }

    // Optimization
    if (single_param)
      parabolic_interpolation_golden_optim<Dtype, ADDITIVE_KERNEL>(
        tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason + tid,
        batch_size, start_season, beta, gamma, optim_alpha, &alpha_, optim_beta,
        &beta_, optim_gamma, &gamma_, optim_params.eps);
    else
      optim = holtwinters_bfgs_optim_device<Dtype, ADDITIVE_KERNEL>(
        tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason + tid,
        batch_size, start_season, beta, gamma, optim_alpha, &alpha_, optim_beta,
        &beta_, optim_gamma, &gamma_, optim_params);

    if (optim_alpha) alpha[tid] = bound_device(alpha_);
    if (optim_beta) beta[tid] = bound_device(beta_);
    if (optim_gamma) gamma[tid] = bound_device(gamma_);
    if (!single_param && optim_result) optim_result[tid] = optim;

    if (error || level || trend || season || xhat) {
      // Final fit
      Dtype error_ = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
        tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason + tid,
        batch_size, start_season, beta, gamma, alpha_, beta_, gamma_, level,
        trend, season, xhat);
      if (error) error[tid] = error_;
    }
  }
}

template <typename Dtype>
__device__ Dtype golden_step(Dtype a, Dtype b, Dtype c) {
  Dtype mid = (a + c) * 0.5;
  if (b > mid)
    return (a - b) * GOLD;
  else
    return (c - b) * GOLD;
}

template <typename Dtype>
__device__ Dtype fix_step(Dtype a, Dtype b, Dtype c, Dtype step, Dtype e) {
  Dtype min_step = ML::HoltWinters::Math::abs_device(e * b) + PG_EPS;
  if (ML::HoltWinters::Math::abs_device(step) < min_step)
    return step > 0 ? min_step : -min_step;
  if (ML::HoltWinters::Math::abs_device(b + step - a) <= e ||
      ML::HoltWinters::Math::abs_device(b + step - c) <= e)
    return 0.0;  // steps are too close to each others
  return step;
}

template <typename Dtype>
__device__ Dtype calculate_step(Dtype a, Dtype b, Dtype c, Dtype loss_a,
                                Dtype loss_b, Dtype loss_c, Dtype pstep,
                                Dtype e) {
  // parabola step
  Dtype p = (b - a) * (loss_b - loss_c);
  Dtype q = (b - c) * (loss_b - loss_a);
  Dtype x = q * (b - c) - p * (b - a);
  Dtype y = (p - q) * 2.;
  Dtype step = ML::HoltWinters::Math::abs_device(y) < PG_EPS
                 ? golden_step(a, b, c)
                 : x / y;
  step = fix_step(a, b, c, step, e);  // ensure point is new

  if (ML::HoltWinters::Math::abs_device(step) >
        ML::HoltWinters::Math::abs_device(pstep / 2) ||
      step == 0.0)
    step = golden_step(a, b, c);
  return step;
}

template <typename Dtype, bool ADDITIVE_KERNEL>
__device__ void parabolic_interpolation_golden_optim(
  int tid, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, Dtype *pseason, int pseason_width,
  const Dtype *start_season, const Dtype *beta, const Dtype *gamma,
  bool optim_alpha, Dtype *alpha_, bool optim_beta, Dtype *beta_,
  bool optim_gamma, Dtype *gamma_, Dtype eps) {
  Dtype a = (Dtype).0;
  Dtype b = (Dtype)GOLD;
  Dtype c = (Dtype)1.;

  Dtype loss_a = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
    tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
    pseason_width, start_season, beta, gamma, optim_alpha ? a : *alpha_,
    optim_beta ? a : *beta_, optim_gamma ? a : *gamma_, nullptr, nullptr,
    nullptr, nullptr);
  Dtype loss_b = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
    tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
    pseason_width, start_season, beta, gamma, optim_alpha ? b : *alpha_,
    optim_beta ? b : *beta_, optim_gamma ? b : *gamma_, nullptr, nullptr,
    nullptr, nullptr);
  Dtype loss_c = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
    tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
    pseason_width, start_season, beta, gamma, optim_alpha ? c : *alpha_,
    optim_beta ? c : *beta_, optim_gamma ? c : *gamma_, nullptr, nullptr,
    nullptr, nullptr);

  Dtype pstep = (c - a) / 2;
  Dtype cstep = pstep;

  while (ML::HoltWinters::Math::abs_device(c - a) >
         ML::HoltWinters::Math::abs_device(b * eps) + PG_EPS) {
    Dtype step = calculate_step(a, b, c, loss_a, loss_b, loss_c, cstep, eps);
    Dtype optim_val = b + step;
    Dtype loss_val = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma,
      optim_alpha ? optim_val : *alpha_, optim_beta ? optim_val : *beta_,
      optim_gamma ? optim_val : *gamma_, nullptr, nullptr, nullptr, nullptr);
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
  if (optim_alpha) *alpha_ = b;
  if (optim_beta) *beta_ = b;
  if (optim_gamma) *gamma_ = b;
}

template <typename Dtype, bool ADDITIVE_KERNEL>
__device__ void holtwinters_finite_gradient_device(
  int tid, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, Dtype *pseason, int pseason_width,
  const Dtype *start_season, const Dtype *beta, const Dtype *gamma,
  Dtype alpha_, Dtype beta_, Dtype gamma_, Dtype *g_alpha, Dtype *g_beta,
  Dtype *g_gamma, Dtype eps) {
  Dtype left_error, right_error;
  if (g_alpha) {  // alpha gradient
    left_error = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma, alpha_ - eps, beta_, gamma_,
      nullptr, nullptr, nullptr, nullptr);
    right_error = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma, alpha_ + eps, beta_, gamma_,
      nullptr, nullptr, nullptr, nullptr);
    *g_alpha = (right_error - left_error) / (eps * 2.);
  }
  if (g_beta) {  // beta gradient
    left_error = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma, alpha_, beta_ - eps, gamma_,
      nullptr, nullptr, nullptr, nullptr);
    right_error = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma, alpha_, beta_ + eps, gamma_,
      nullptr, nullptr, nullptr, nullptr);
    *g_beta = (right_error - left_error) / (eps * 2.);
  }
  if (g_gamma) {  // gamma gradient
    left_error = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma, alpha_, beta_, gamma_ - eps,
      nullptr, nullptr, nullptr, nullptr);
    right_error = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma, alpha_, beta_, gamma_ + eps,
      nullptr, nullptr, nullptr, nullptr);
    *g_gamma = (right_error - left_error) / (eps * 2.);
  }
}

template <typename Dtype, bool ADDITIVE_KERNEL>
__device__ ML::OptimCriterion holtwinters_bfgs_optim_device(
  int tid, const Dtype *ts, int n, int batch_size, int frequency, int shift,
  Dtype plevel, Dtype ptrend, Dtype *pseason, int pseason_width,
  const Dtype *start_season, const Dtype *beta, const Dtype *gamma,
  bool optim_alpha, Dtype *x1, bool optim_beta, Dtype *x2, bool optim_gamma,
  Dtype *x3, const ML::OptimParams<Dtype> optim_params) {
  Dtype H11 = 1., H12 = .0, H13 = .0, H22 = 1., H23 = .0,
        H33 = 1.;  // Hessian approximiation (Hessian is symmetric)
  Dtype g1 = .0, g2 = .0, g3 = .0;  // gradients

  // initial gradient
  holtwinters_finite_gradient_device<Dtype, ADDITIVE_KERNEL>(
    tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
    pseason_width, start_season, beta, gamma, *x1, *x2, *x3,
    optim_alpha ? &g1 : nullptr, optim_beta ? &g2 : nullptr,
    optim_gamma ? &g3 : nullptr, optim_params.eps);

  for (int iter = 0; iter < optim_params.bfgs_iter_limit; ++iter) {
    // Step direction
    Dtype p1 = -H11 * g1 - H12 * g2 - H13 * g3;
    Dtype p2 = -H12 * g1 - H22 * g2 - H23 * g3;
    Dtype p3 = -H13 * g1 - H23 * g2 - H33 * g3;

    const Dtype phi = p1 * g1 + p2 * g2 + p3 * g3;
    if (phi > 0) {
      H11 = 1.;
      H12 = 0.;
      H13 = 0.;
      H22 = 1.;
      H23 = 0.;
      H33 = 1.;
      p1 = -g1;
      p2 = -g2;
      p3 = -g3;
    }

    // {next_params} = {params}+step_size*p;
    // start of line search

    // starting step size, we assume the largest distance between x and nx is going to be sqrt(3)/2. where sqrt(3)
    // is the largest allowed step in a 1x1x1 cube.
    Dtype step_size;
    if (optim_params.linesearch_step_size <= 0)
      step_size = (Dtype)0.866 / sqrt(p1 * p1 + p2 * p2 + p3 * p3);
    else
      step_size = optim_params.linesearch_step_size;
    Dtype nx1 = *x1 + step_size * p1;
    Dtype nx2 = *x2 + step_size * p2;
    Dtype nx3 = *x3 + step_size * p3;

    // line search params
    const Dtype cauchy =
      optim_params.linesearch_c * (g1 * p1 + g2 * p2 + g3 * p3);
    const Dtype loss_ref = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma, *x1, *x2, *x3, nullptr, nullptr,
      nullptr, nullptr);
    Dtype loss = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma, nx1, nx2, nx3, nullptr, nullptr,
      nullptr, nullptr);

    for (int i = 0; i < optim_params.linesearch_iter_limit &&
                    (loss > loss_ref + step_size * cauchy);
         ++i) {
      step_size *= optim_params.linesearch_tau;
      nx1 = *x1 + step_size * p1;
      nx2 = *x2 + step_size * p2;
      nx3 = *x3 + step_size * p3;
      loss = holtwinters_eval_device<Dtype, ADDITIVE_KERNEL>(
        tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
        pseason_width, start_season, beta, gamma, nx1, nx2, nx3, nullptr,
        nullptr, nullptr, nullptr);
    }
    // end of line search

    // see if new {prams} meet stop condition
    const Dtype dx1 = ML::HoltWinters::Math::abs_device(*x1 - nx1);
    const Dtype dx2 = ML::HoltWinters::Math::abs_device(*x2 - nx2);
    const Dtype dx3 = ML::HoltWinters::Math::abs_device(*x3 - nx3);
    Dtype max = max3(dx1, dx2, dx3);
    // update {params}
    *x1 = nx1;
    *x2 = nx2;
    *x3 = nx3;
    if (optim_params.min_param_diff > max)
      return ML::OptimCriterion::OPTIM_MIN_PARAM_DIFF;
    if (optim_params.min_error_diff >
        ML::HoltWinters::Math::abs_device(loss - loss_ref))
      return ML::OptimCriterion::OPTIM_MIN_ERROR_DIFF;

    Dtype ng1 = .0, ng2 = .0, ng3 = .0;  // next gradient
    holtwinters_finite_gradient_device<Dtype, ADDITIVE_KERNEL>(
      tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
      pseason_width, start_season, beta, gamma, nx1, nx2, nx3,
      optim_alpha ? &ng1 : nullptr, optim_beta ? &ng2 : nullptr,
      optim_gamma ? &ng3 : nullptr, optim_params.eps);
    // see if new gradients meet stop condition
    max = max3(ML::HoltWinters::Math::abs_device(ng1),
               ML::HoltWinters::Math::abs_device(ng2),
               ML::HoltWinters::Math::abs_device(ng3));
    if (optim_params.min_grad_norm > max)
      return ML::OptimCriterion::OPTIM_MIN_GRAD_NORM;

    // s = step_size*p;
    const Dtype s1 = step_size * p1;
    const Dtype s2 = step_size * p2;
    const Dtype s3 = step_size * p3;

    // y = next_grad-grad
    const Dtype y1 = ng1 - g1;
    const Dtype y2 = ng2 - g2;
    const Dtype y3 = ng3 - g3;

    // rho_ = y(*)s; rho = 1/rho_
    const Dtype rho_ = y1 * s1 + y2 * s2 + y3 * s3;
    const Dtype rho = 1.0 / rho_;

    const Dtype Hy1 = H11 * y1 + H12 * y2 + H13 * y3;
    const Dtype Hy2 = H12 * y1 + H22 * y2 + H23 * y3;
    const Dtype Hy3 = H13 * y1 + H23 * y2 + H33 * y3;
    const Dtype k = rho * rho * (y1 * Hy1 + y2 * Hy2 + y3 * Hy3 + rho_);

    H11 += k * s1 * s1 - 2. * rho * s1 * Hy1;
    H12 += k * s1 * s2 - rho * (s2 * Hy1 + s1 * Hy2);
    H13 += k * s1 * s3 - rho * (s3 * Hy1 + s1 * Hy3);
    H22 += k * s2 * s2 - 2 * rho * s2 * Hy2;
    H23 += k * s2 * s3 - rho * (s3 * Hy2 + s2 * Hy3);
    H33 += k * s3 * s3 - 2. * rho * s3 * Hy3;

    g1 = ng1;
    g2 = ng2;
    g3 = ng3;
  }

  return ML::OptimCriterion::OPTIM_BFGS_ITER_LIMIT;
}