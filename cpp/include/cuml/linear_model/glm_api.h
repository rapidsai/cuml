/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/cuml_api.h>
#include <cuml/linear_model/qn.h>

#include <stdbool.h>

#ifdef __cplusplus
namespace ML::GLM {
extern "C" {
#endif

cumlError_t cumlSpQnFit(cumlHandle_t cuml_handle,
                        const qn_params* pams,
                        float* X,
                        float* y,
                        int N,
                        int D,
                        int C,
                        float* w0,
                        float* f,
                        int* num_iters,
                        bool X_col_major);

cumlError_t cumlDpQnFit(cumlHandle_t cuml_handle,
                        const qn_params* pams,
                        double* X,
                        double* y,
                        int N,
                        int D,
                        int C,
                        double* w0,
                        double* f,
                        int* num_iters,
                        bool X_col_major);

#ifdef __cplusplus
}
}
#endif
