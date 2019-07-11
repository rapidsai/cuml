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

#define HW_SAFE_CALL(call)                                            \
  {                                                                   \
    do {                                                              \
      ML::HWStatus status = call;                                     \
      if (status != ML::HWStatus::HW_SUCCESS) {                       \
        std::cerr << "HW error in in line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                           \
      }                                                               \
    } while (0);                                                      \
  }

#define CUDA_SAFE_CALL(call)                                               \
  {                                                                        \
    do {                                                                   \
      cudaError_t status = call;                                           \
      ASSERT(status == cudaSuccess, "FAIL: call='%s'. Reason:%s\n", #call, \
             cudaGetErrorString(status));                                  \
    } while (0);                                                           \
  }

namespace ML {

enum HWStatus {
  HW_SUCCESS = 0,
  HW_NOT_INITIALIZED = 1,
  HW_INVALID_VALUE = 2,
  HW_ALLOC_FAILED = 3,
  HW_INTERNAL_ERROR = 4
};

enum SeasonalType { ADDITIVE, MULTIPLICATIVE };

enum OptimCriterion {
  OPTIM_BFGS_ITER_LIMIT = 0,
  OPTIM_MIN_PARAM_DIFF = 1,
  OPTIM_MIN_ERROR_DIFF = 2,
  OPTIM_MIN_GRAD_NORM = 3,
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

enum Norm { L0, L1, L2, LINF };

// HW misc functions
HWStatus HWInit();
HWStatus HWDestroy();

template <typename Dtype>
HWStatus HWTranspose(const Dtype *data_in, int m, int n, Dtype *data_out);

HWStatus HoltWintersBufferSize(int n, int batch_size, int frequency,
                               bool use_beta, bool use_gamma,
                               int *start_leveltrend_len, int *start_season_len,
                               int *components_len, int *error_len,
                               int *leveltrend_coef_shift,
                               int *season_coef_shift);

template <typename Dtype>
HWStatus HoltWintersDecompose(const Dtype *ts, int n, int batch_size,
                              int frequency, Dtype *start_level,
                              Dtype *start_trend, Dtype *start_season,
                              int start_periods,
                              SeasonalType seasonal = ADDITIVE);

template <typename Dtype>
HWStatus HoltWintersOptim(const Dtype *ts, int n, int batch_size, int frequency,
                          const Dtype *start_level, const Dtype *start_trend,
                          const Dtype *start_season, Dtype *alpha,
                          bool optim_alpha, Dtype *beta, bool optim_beta,
                          Dtype *gamma, bool optim_gamma, Dtype *level,
                          Dtype *trend, Dtype *season, Dtype *xhat,
                          Dtype *error, OptimCriterion *optim_result,
                          OptimParams<Dtype> *optim_params = nullptr,
                          SeasonalType seasonal = ADDITIVE);

template <typename Dtype>
HWStatus HoltWintersEval(const Dtype *ts, int n, int batch_size, int frequency,
                         const Dtype *start_level, const Dtype *start_trend,
                         const Dtype *start_season, const Dtype *alpha,
                         const Dtype *beta, const Dtype *gamma, Dtype *level,
                         Dtype *trend, Dtype *season, Dtype *xhat, Dtype *error,
                         SeasonalType seasonal = ADDITIVE);

template <typename Dtype>
HWStatus HoltWintersForecast(Dtype *forecast, int h, int batch_size,
                             int frequency, const Dtype *level_coef,
                             const Dtype *trend_coef, const Dtype *season_coef,
                             SeasonalType seasonal = ADDITIVE);

template <typename Dtype>
void HoltWintersFitPredict(int n, int batch_size, int frequency, int h,
                           int start_periods, SeasonalType seasonal,
                           Dtype *data, Dtype *alpha_ptr, Dtype *beta_ptr,
                           Dtype *gamma_ptr, Dtype *SSE_error_ptr,
                           Dtype *forecast_ptr);

}  // namespace ML
