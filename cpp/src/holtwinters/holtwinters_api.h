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

#include <cuML_api.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cumlHoltWintersSeasonal_t {
    ADDITIVE, MULTIPLICATIVE
};

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
             * @return CUML_SUCCESS on success and other corresponding flags upon any failures.
             * @{
  */
cumlError_t cuml_buffer_size(int n, int batch_size, int frequency,
                 int *start_leveltrend_len, int *start_season_len,
                 int *components_len, int *error_len,
                 int *leveltrend_coef_shift, int *season_coef_shift);
/** @} */

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
             * @param level_ptr
             *        host pointer to array which will hold level components
             * @param trend_ptr
             *        host pointer to array which will hold trend components
             * @param season_ptr
             *        host pointer to array which will hold season components
             * @param SSE_error_ptr
             *        host pointer to array which will hold training SSE error
             * @return CUML_SUCCESS on success and other corresponding flags upon any failures.
             * @{
  */
cumlError_t cumlSp_fit(cumlHandle_t handle, int n, int batch_size, int frequency,
         int start_periods, cumlHoltWintersSeasonal_t seasonal, float *data,
         float *level_ptr, float *trend_ptr, float *season_ptr,
         float *SSE_error_ptr);
cumlError_t cumlDp_fit(cumlHandle_t handle, int n, int batch_size, int frequency,
         int start_periods, cumlHoltWintersSeasonal_t seasonal, double *data,
         double *level_ptr, double *trend_ptr, double *season_ptr,
         double *SSE_error_ptr);
/** @} */

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
             * @param level_ptr
             *        host pointer to array which holds level components
             * @param trend_ptr
             *        host pointer to array which holds trend components
             * @param season_ptr
             *        host pointer to array which holds season components
             * @param forecast_ptr
             *        host pointer to array which will hold the predicted points
             * @return CUML_SUCCESS on success and other corresponding flags upon any failures.
             * @{
  */
cumlError_t cumlSp_predict(cumlHandle_t handle, int n, int batch_size, int frequency,
             int h, cumlHoltWintersSeasonal_t seasonal, float *level_ptr, float *trend_ptr,
             float *season_ptr, float *forecast_ptr);
cumlError_t cumlDp_predict(cumlHandle_t handle, int n, int batch_size, int frequency,
             int h, cumlHoltWintersSeasonal_t seasonal, double *level_ptr, double *trend_ptr,
             double *season_ptr, double *forecast_ptr);
/** @} */

#ifdef __cplusplus
}
#endif
