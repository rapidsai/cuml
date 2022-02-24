/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

namespace raft {
class handle_t;
}

namespace ML {
namespace HoltWinters {

/**
 * Provides buffer sizes for HoltWinters algorithm
 * @param[in] n
 *            n_samples in time-series
 * @param[in] batch_size
 *            number of time-series in X
 * @param[in] frequency
 *            number of periods in a season of the time-series
 * @param[out] start_leveltrend_len
 *             pointer which will hold the length of the level/trend array buffers
 * @param[out] start_season_len
 *             pointer which will hold the length of the seasonal array buffer
 * @param[out] components_len
 *             pointer which will hold the length of all three components
 * @param[out] error_len
 *             pointer which will hold the length of the SSE Error
 * @param[out] leveltrend_coef_shift
 *             pointer which will hold the offset to level/trend arrays
 * @param[out] season_coef_shift
 *             pointer which will hold the offset to season array
 */
void buffer_size(int n,
                 int batch_size,
                 int frequency,
                 int* start_leveltrend_len,
                 int* start_season_len,
                 int* components_len,
                 int* error_len,
                 int* leveltrend_coef_shift,
                 int* season_coef_shift);

/**
 * Fits a HoltWinters model
 * @param[in] handle
 *            cuml handle to use across the algorithm
 * @param[in] n
 *            n_samples in time-series
 * @param[in] batch_size
 *            number of time-series in X
 * @param[in] frequency
 *            number of periods in a season of the time-series
 * @param[in] start_periods
 *            number of seasons to be used for seasonal seed values
 * @param[in] seasonal
 *            type of seasonal component (ADDITIVE or MULTIPLICATIVE)
 * @param[in] epsilon
 *            the error tolerance value for optimization
 * @param[in] data
 *            device pointer to the data to fit on
 * @param[out] level_d
 *             device pointer to array which will hold level components
 * @param[out] trend_d
 *             device pointer to array which will hold trend components
 * @param[out] season_d
 *             device pointer to array which will hold season components
 * @param[out] error_d
 *             device pointer to array which will hold training SSE error
 */
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
         float* error_d);
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
         double* error_d);

/**
 * Forecasts future points from fitted HoltWinters model
 * @param[in] handle
 *            cuml handle to use across the algorithm
 * @param[in] n
 *            n_samples in time-series
 * @param[in] batch_size
 *            number of time-series in X
 * @param[in] frequency
 *            number of periods in a season of the time-series
 * @param[in] h
 *            number of future points to predict in the time-series
 * @param[in] seasonal
 *            type of seasonal component (ADDITIVE or MULTIPLICATIVE)
 * @param[out] level_d
 *             device pointer to array which holds level components
 * @param[out] trend_d
 *             device pointer to array which holds trend components
 * @param[out] season_d
 *             device pointer to array which holds season components
 * @param[out] forecast_d
 *             device pointer to array which will hold the forecast points
 */
void forecast(const raft::handle_t& handle,
              int n,
              int batch_size,
              int frequency,
              int h,
              ML::SeasonalType seasonal,
              float* level_d,
              float* trend_d,
              float* season_d,
              float* forecast_d);
void forecast(const raft::handle_t& handle,
              int n,
              int batch_size,
              int frequency,
              int h,
              ML::SeasonalType seasonal,
              double* level_d,
              double* trend_d,
              double* season_d,
              double* forecast_d);

}  // namespace HoltWinters
}  // namespace ML
