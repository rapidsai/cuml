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

cumlError_t cuml_buffer_size(int n, int batch_size, int frequency,
                 int *start_leveltrend_len, int *start_season_len,
                 int *components_len, int *error_len,
                 int *leveltrend_coef_shift, int *season_coef_shift);

cumlError_t cumlSp_fit(cumlHandle_t handle, int n, int batch_size, int frequency,
         int start_periods, cumlHoltWintersSeasonal_t seasonal, float *data,
         float *level_ptr, float *trend_ptr, float *season_ptr,
         float *SSE_error_ptr);

cumlError_t cumlDp_fit(cumlHandle_t handle, int n, int batch_size, int frequency,
         int start_periods, cumlHoltWintersSeasonal_t seasonal, double *data,
         double *level_ptr, double *trend_ptr, double *season_ptr,
         double *SSE_error_ptr);

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
