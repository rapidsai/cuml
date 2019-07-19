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

#include "holtwinters.h"
#include "runner.h"

namespace ML {
namespace HoltWinters {

void buffer_size(int n, int batch_size, int frequency,
                 int *start_leveltrend_len, int *start_season_len,
                 int *components_len, int *error_len,
                 int *leveltrend_coef_shift, int *season_coef_shift) {
  bool use_beta = true, use_gamma = true;
  ML::HoltWintersBufferSize(n, batch_size, frequency, use_beta, use_gamma,
                            start_leveltrend_len, start_season_len,
                            components_len, error_len, leveltrend_coef_shift,
                            season_coef_shift);
}

void fit(const ML::cumlHandle &handle, int n, int batch_size, int frequency,
         int start_periods, ML::SeasonalType seasonal, float *data,
         float *level_ptr, float *trend_ptr, float *season_ptr,
         float *SSE_error_ptr) {
  ML::HoltWintersFit<float>(handle, n, batch_size, frequency, start_periods,
                            seasonal, data, level_ptr, trend_ptr, season_ptr,
                            SSE_error_ptr);
}

void fit(const ML::cumlHandle &handle, int n, int batch_size, int frequency,
         int start_periods, ML::SeasonalType seasonal, double *data,
         double *level_ptr, double *trend_ptr, double *season_ptr,
         double *SSE_error_ptr) {
  ML::HoltWintersFit<double>(handle, n, batch_size, frequency, start_periods,
                             seasonal, data, level_ptr, trend_ptr, season_ptr,
                             SSE_error_ptr);
}

void predict(const ML::cumlHandle &handle, int n, int batch_size, int frequency,
             int h, ML::SeasonalType seasonal, float *level_ptr,
             float *trend_ptr, float *season_ptr, float *forecast_ptr) {
  ML::HoltWintersPredict<float>(handle, n, batch_size, frequency, h, seasonal,
                                level_ptr, trend_ptr, season_ptr, forecast_ptr);
}

void predict(const ML::cumlHandle &handle, int n, int batch_size, int frequency,
             int h, ML::SeasonalType seasonal, double *level_ptr,
             double *trend_ptr, double *season_ptr, double *forecast_ptr) {
  ML::HoltWintersPredict<double>(handle, n, batch_size, frequency, h, seasonal,
                                 level_ptr, trend_ptr, season_ptr,
                                 forecast_ptr);
}

}  // namespace HoltWinters
}  // namespace ML