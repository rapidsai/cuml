/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/cumlHandle.hpp>

#include <cuml/tsa/holtwinters.h>
#include <cuml/tsa/holtwinters_api.h>

extern "C" {

cumlError_t cumlHoltWinters_buffer_size(int n,
                                        int batch_size,
                                        int frequency,
                                        int* start_leveltrend_len,
                                        int* start_season_len,
                                        int* components_len,
                                        int* error_len,
                                        int* leveltrend_coef_shift,
                                        int* season_coef_shift)
{
  cumlError_t status;
  try {
    ML::HoltWinters::buffer_size(n,
                                 batch_size,
                                 frequency,
                                 start_leveltrend_len,
                                 start_season_len,
                                 components_len,
                                 error_len,
                                 leveltrend_coef_shift,
                                 season_coef_shift);
    status = CUML_SUCCESS;
  } catch (...) {
    status = CUML_ERROR_UNKNOWN;
  }
  return status;
}

cumlError_t cumlHoltWintersSp_fit(cumlHandle_t handle,
                                  int n,
                                  int batch_size,
                                  int frequency,
                                  int start_periods,
                                  cumlHoltWintersSeasonal_t seasonal,
                                  float epsilon,
                                  float* data,
                                  float* level_d,
                                  float* trend_d,
                                  float* season_d,
                                  float* error_d)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SeasonalType seasonal_type = (ML::SeasonalType)seasonal;
      ML::HoltWinters::fit(*handle_ptr,
                           n,
                           batch_size,
                           frequency,
                           start_periods,
                           seasonal_type,
                           epsilon,
                           data,
                           level_d,
                           trend_d,
                           season_d,
                           error_d);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlHoltWintersDp_fit(cumlHandle_t handle,
                                  int n,
                                  int batch_size,
                                  int frequency,
                                  int start_periods,
                                  cumlHoltWintersSeasonal_t seasonal,
                                  double epsilon,
                                  double* data,
                                  double* level_d,
                                  double* trend_d,
                                  double* season_d,
                                  double* error_d)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SeasonalType seasonal_type = (ML::SeasonalType)seasonal;
      ML::HoltWinters::fit(*handle_ptr,
                           n,
                           batch_size,
                           frequency,
                           start_periods,
                           seasonal_type,
                           epsilon,
                           data,
                           level_d,
                           trend_d,
                           season_d,
                           error_d);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlHoltWintersSp_forecast(cumlHandle_t handle,
                                       int n,
                                       int batch_size,
                                       int frequency,
                                       int h,
                                       cumlHoltWintersSeasonal_t seasonal,
                                       float* level_d,
                                       float* trend_d,
                                       float* season_d,
                                       float* forecast_d)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SeasonalType seasonal_type = (ML::SeasonalType)seasonal;
      ML::HoltWinters::forecast(*handle_ptr,
                                n,
                                batch_size,
                                frequency,
                                h,
                                seasonal_type,
                                level_d,
                                trend_d,
                                season_d,
                                forecast_d);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlHoltWintersDp_forecast(cumlHandle_t handle,
                                       int n,
                                       int batch_size,
                                       int frequency,
                                       int h,
                                       cumlHoltWintersSeasonal_t seasonal,
                                       double* level_d,
                                       double* trend_d,
                                       double* season_d,
                                       double* forecast_d)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SeasonalType seasonal_type = (ML::SeasonalType)seasonal;
      ML::HoltWinters::forecast(*handle_ptr,
                                n,
                                batch_size,
                                frequency,
                                h,
                                seasonal_type,
                                level_d,
                                trend_d,
                                season_d,
                                forecast_d);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}
}
