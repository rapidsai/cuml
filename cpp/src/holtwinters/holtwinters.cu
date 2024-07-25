/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "runner.cuh"

#include <cuml/tsa/holtwinters.h>

namespace ML {
namespace HoltWinters {

void buffer_size(int n,
                 int batch_size,
                 int frequency,
                 int* start_leveltrend_len,
                 int* start_season_len,
                 int* components_len,
                 int* error_len,
                 int* leveltrend_coef_shift,
                 int* season_coef_shift)
{
  bool use_beta = true, use_gamma = true;
  ML::HoltWintersBufferSize(n,
                            batch_size,
                            frequency,
                            use_beta,
                            use_gamma,
                            start_leveltrend_len,
                            start_season_len,
                            components_len,
                            error_len,
                            leveltrend_coef_shift,
                            season_coef_shift);
}

void fit(const raft::handle_t& handle,
         int n,
         int batch_size,
         int frequency,
         int start_periods,
         ML::SeasonalType seasonal,
         float epsilon,
         float* data,
         float* level_d,
         float* trend_d,
         float* season_d,
         float* error_d)
{
  ML::HoltWintersFitHelper<float>(handle,
                                  n,
                                  batch_size,
                                  frequency,
                                  start_periods,
                                  seasonal,
                                  epsilon,
                                  data,
                                  level_d,
                                  trend_d,
                                  season_d,
                                  error_d);
}

void fit(const raft::handle_t& handle,
         int n,
         int batch_size,
         int frequency,
         int start_periods,
         ML::SeasonalType seasonal,
         double epsilon,
         double* data,
         double* level_d,
         double* trend_d,
         double* season_d,
         double* error_d)
{
  ML::HoltWintersFitHelper<double>(handle,
                                   n,
                                   batch_size,
                                   frequency,
                                   start_periods,
                                   seasonal,
                                   epsilon,
                                   data,
                                   level_d,
                                   trend_d,
                                   season_d,
                                   error_d);
}

void forecast(const raft::handle_t& handle,
              int n,
              int batch_size,
              int frequency,
              int h,
              ML::SeasonalType seasonal,
              float* level_d,
              float* trend_d,
              float* season_d,
              float* forecast_d)
{
  ML::HoltWintersForecastHelper<float>(
    handle, n, batch_size, frequency, h, seasonal, level_d, trend_d, season_d, forecast_d);
}

void forecast(const raft::handle_t& handle,
              int n,
              int batch_size,
              int frequency,
              int h,
              ML::SeasonalType seasonal,
              double* level_d,
              double* trend_d,
              double* season_d,
              double* forecast_d)
{
  ML::HoltWintersForecastHelper<double>(
    handle, n, batch_size, frequency, h, seasonal, level_d, trend_d, season_d, forecast_d);
}

}  // namespace HoltWinters
}  // namespace ML
