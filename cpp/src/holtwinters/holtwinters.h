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
#include "cuML.hpp"
#include "holtwinters_params.h"

namespace ML {
namespace HoltWinters {

/**
             * Provifes buffer sizes for HoltWinters algorithm
             * @param n
             *        n_samples in time-series
             * @param batch_size
             *        number of time-series in X
             * @param frequency
             *        number of periods in a season of the time-series
             * @param start_leveltrend_len
             *        pointer which will hold the length of the level/trend array buffers
             * @param start_season_len
             *        pointer which will hold the length of the seasonal array buffer
             * @param components_len
             *        pointer which will hold the length of all three components
             * @param error_len
             *        pointer which will hold the length of the SSE Error
             * @param leveltrend_coef_shift
             *        pointer which will hold the offset to level/trend arrays
             * @param season_coef_shift
             *        pointer which will hold the offset to season array
             */
void buffer_size(int n, int batch_size, int frequency,
                 int *start_leveltrend_len, int *start_season_len,
                 int *components_len, int *error_len,
                 int *leveltrend_coef_shift, int *season_coef_shift);

/**
             * Fits a HoltWinters model
             * @param handle
             *        cuml handle to use across the algorithm
             * @param n
             *        n_samples in time-series
             * @param batch_size
             *        number of time-series in X
             * @param frequency
             *        number of periods in a season of the time-series
             * @param start_periods
             *        number of seasons to be used for seasonal seed values
             * @param seasonal
             *        type of seasonal component (ADDITIVE or MULTIPLICATIVE)
             * @param data
             *        device pointer to the data to fit on
             * @param level_d
             *        device pointer to array which will hold level components
             * @param trend_d
             *        device pointer to array which will hold trend components
             * @param season_d
             *        device pointer to array which will hold season components
             * @param error_d
             *        device pointer to array which will hold training SSE error
             */
void fit(const ML::cumlHandle &handle, int n, int batch_size, int frequency,
         int start_periods, ML::SeasonalType seasonal, float *data,
         float *level_d, float *trend_d, float *season_d, float *error_d);
void fit(const ML::cumlHandle &handle, int n, int batch_size, int frequency,
         int start_periods, ML::SeasonalType seasonal, double *data,
         double *level_d, double *trend_d, double *season_d, double *error_d);

/**
             * Fits a HoltWinters model
             * @param handle
             *        cuml handle to use across the algorithm
             * @param n
             *        n_samples in time-series
             * @param batch_size
             *        number of time-series in X
             * @param frequency
             *        number of periods in a season of the time-series
             * @param h
             *        number of future points to predict in the time-series
             * @param seasonal
             *        type of seasonal component (ADDITIVE or MULTIPLICATIVE)
             * @param level_d
             *        device pointer to array which holds level components
             * @param trend_d
             *        device pointer to array which holds trend components
             * @param season_d
             *        device pointer to array which holds season components
             * @param forecast_d
             *        device pointer to array which will hold the predicted points
             */
void predict(const ML::cumlHandle &handle, int n, int batch_size, int frequency,
             int h, ML::SeasonalType seasonal, float *level_d, float *trend_d,
             float *season_d, float *forecast_d);
void predict(const ML::cumlHandle &handle, int n, int batch_size, int frequency,
             int h, ML::SeasonalType seasonal, double *level_d, double *trend_d,
             double *season_d, double *forecast_d);

}  // namespace HoltWinters
}  // namespace ML