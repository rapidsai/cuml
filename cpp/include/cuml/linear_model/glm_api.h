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

#include <cuml/cuml_api.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

cumlError_t cumlSpQnFit(cumlHandle_t cuml_handle, float *X, float *y, int N,
                        int D, int C, bool fit_intercept, float l1, float l2,
                        int max_iter, float grad_tol, int linesearch_max_iter,
                        int lbfgs_memory, int verbosity, float *w0, float *f,
                        int *num_iters, bool X_col_major, int loss_type);

cumlError_t cumlDpQnFit(cumlHandle_t cuml_handle, double *X, double *y, int N,
                        int D, int C, bool fit_intercept, double l1, double l2,
                        int max_iter, double grad_tol, int linesearch_max_iter,
                        int lbfgs_memory, int verbosity, double *w0, double *f,
                        int *num_iters, bool X_col_major, int loss_type);

#ifdef __cplusplus
}
#endif
