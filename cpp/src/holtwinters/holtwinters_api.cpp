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
#include "holtwinters_api.h"
#include <cuML_api.h>
#include "common/cumlHandle.hpp"
#include "holtwinters.h"

cumlError_t cumlHoltWinters_buffer_size(int n, int batch_size, int frequency,
                                        int *start_leveltrend_len,
                                        int *start_season_len,
                                        int *components_len, int *error_len,
                                        int *leveltrend_coef_shift,
                                        int *season_coef_shift) {
  cumlError_t status;
  if (status == CUML_SUCCESS) {
    try {
      ML::HoltWinters::buffer_size(
        n, batch_size, frequency, start_leveltrend_len, start_season_len,
        components_len, error_len, leveltrend_coef_shift, season_coef_shift);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlcumlHoltWintersSp_fit(
  cumlHandle_t handle, int n, int batch_size, int frequency, int start_periods,
  cumlHoltWintersSeasonal_t seasonal, float *data, float *level_d,
  float *trend_d, float *season_d, float *error_d) {
  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SeasonalType seasonal_type = (ML::SeasonalType)seasonal;
      ML::HoltWinters::fit(*handle_ptr, n, batch_size, frequency, start_periods,
                           seasonal_type, data, level_d, trend_d, season_d,
                           error_d);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlcumlHoltWintersDp_fit(
  cumlHandle_t handle, int n, int batch_size, int frequency, int start_periods,
  cumlHoltWintersSeasonal_t seasonal, double *data, double *level_d,
  double *trend_d, double *season_d, double *error_d) {
  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SeasonalType seasonal_type = (ML::SeasonalType)seasonal;
      ML::HoltWinters::fit(*handle_ptr, n, batch_size, frequency, start_periods,
                           seasonal_type, data, level_d, trend_d, season_d,
                           error_d);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlcumlHoltWintersSp_forecast(cumlHandle_t handle, int n,
                                           int batch_size, int frequency, int h,
                                           cumlHoltWintersSeasonal_t seasonal,
                                           float *level_d, float *trend_d,
                                           float *season_d, float *forecast_d) {
  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SeasonalType seasonal_type = (ML::SeasonalType)seasonal;
      ML::HoltWinters::forecast(*handle_ptr, n, batch_size, frequency, h,
                                seasonal_type, level_d, trend_d, season_d,
                                forecast_d);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlcumlHoltWintersDp_forecast(cumlHandle_t handle, int n,
                                           int batch_size, int frequency, int h,
                                           cumlHoltWintersSeasonal_t seasonal,
                                           double *level_d, double *trend_d,
                                           double *season_d,
                                           double *forecast_d) {
  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SeasonalType seasonal_type = (ML::SeasonalType)seasonal;
      ML::HoltWinters::forecast(*handle_ptr, n, batch_size, frequency, h,
                                seasonal_type, level_d, trend_d, season_d,
                                forecast_d);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}