/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cumlSvmKernelType { LINEAR, POLYNOMIAL, RBF, TANH } cumlSvmKernelType;

/**
 * @defgroup SVM C-wrapper to C++ implementation of Support Vector Machine
 *
 * The output device buffers shall be unallocated on entry.
 * Note that n_support, n_classes and b are host scalars, all other output
 * pointers are device pointers.
 *
 * @param [in] handle the cuML handle
 * @param [in] input device pointer for the input data in column major format.
 *   Size n_rows x n_cols.
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of columns
 * @param [in] labels device pointer for the labels. Size n_rows.
 * @param [in] C penalty term
 * @param [in] cache_size size of kernel cache in device memory (MiB)
 * @param [in] max_iter maximum number of outer iterations in SmoSolver
 * @param [in] nochange_steps max number of outer iterations without change in convergence
 * @param [in] tol tolerance to stop fitting
 * @param [in] verbosity Fine grained control over logging of useful information
 *   as algorithm executes. Currently passing anything greater than or equal to
 *   rapids_logger::level_enum::info will make it execute quietly
 * @param [in] kernel type of kernel (LINEAR, POLYNOMIAL, RBF or TANH)
 * @param [in] degree of polynomial kernel (ignored by others)
 * @param [in] gamma multiplier in the RBF, POLYNOMIAL and TANH kernels
 * @param [in] coef0 additive constant in poly and tanh kernels
 * @param [out] n_support number of support vectors
 * @param [out] b constant used in the decision function
 * @param [out] dual_coefs non-zero dual coefficients, size [n_support].
 * @param [out] x_support support vectors in column major format.
 *    Size [n_support x n_cols].
 * @param [out] support_idx indices (from the traning set) of the support
 *    vectors, size [n_support].
 * @param [out] n_classes number of classes found in the input labels
 * @param [out] unique_labels device pointer for the unique classes,
 *    size [n_classes]
 * @return CUML_SUCCESS on success and other corresponding flags upon any failures.
 * @{
 */
cumlError_t cumlSpSvcFit(cumlHandle_t handle,
                         float* input,
                         int n_rows,
                         int n_cols,
                         float* labels,
                         float C,
                         float cache_size,
                         int max_iter,
                         int nochange_steps,
                         float tol,
                         int verbosity,
                         cumlSvmKernelType kernel,
                         int degree,
                         float gamma,
                         float coef0,
                         int* n_support,
                         float* b,
                         float** dual_coefs,
                         float** x_support,
                         int** support_idx,
                         int* n_classes,
                         float** unique_labels);

cumlError_t cumlDpSvcFit(cumlHandle_t handle,
                         double* input,
                         int n_rows,
                         int n_cols,
                         double* labels,
                         double C,
                         double cache_size,
                         int max_iter,
                         int nochange_steps,
                         double tol,
                         int verbosity,
                         cumlSvmKernelType kernel,
                         int degree,
                         double gamma,
                         double coef0,
                         int* n_support,
                         double* b,
                         double** dual_coefs,
                         double** x_support,
                         int** support_idx,
                         int* n_classes,
                         double** unique_labels);
/** @} */

/**
 * @defgroup SVM C-wrapper to C++ implementation of Support Vector Machine
 *
 * The output preds array shall be allocated on entry.
 *
 * @param [in] handle the cuML handle
 * @param [in] input device pointer for the input data in column major format.
 *   Size n_rows x n_cols.
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of columns
 * @param [in] kernel type of kernel (LINEAR, POLYNOMIAL, RBF or TANH)
 * @param [in] degree of polynomial kernel (ignored by others)
 * @param [in] gamma multiplier in the RBF, POLYNOMIAL and TANH kernels
 * @param [in] coef0 additive constant in poly and tanh kernels
 * @param [in] n_support number of support vectors
 * @param [in] b constant used in the decision function
 * @param [in] dual_coefs non-zero dual coefficients, size [n_support].
 * @param [in] x_support support vectors in column major format.
 *    Size [n_support x n_cols].
 * @param [in] n_classes number of classes found in the input labels
 * @param [in] unique_labels device pointer for the unique classes,
 *    size [n_classes]
 * @param [out] preds device pointer for the predictions. Size [n_rows].
 * @param [in] buffer_size size of temporary buffer in MiB
 * @param [in] predict_class whether to predict class label (true), or just
 *     return the decision function value (false)
 * @return CUML_SUCCESS on success and other corresponding flags upon any failures.
 * @{
 */
cumlError_t cumlSpSvcPredict(cumlHandle_t handle,
                             float* input,
                             int n_rows,
                             int n_cols,
                             cumlSvmKernelType kernel,
                             int degree,
                             float gamma,
                             float coef0,
                             int n_support,
                             float b,
                             float* dual_coefs,
                             float* x_support,
                             int n_classes,
                             float* unique_labels,
                             float* preds,
                             float buffer_size,
                             int predict_class);

cumlError_t cumlDpSvcPredict(cumlHandle_t handle,
                             double* input,
                             int n_rows,
                             int n_cols,
                             cumlSvmKernelType kernel,
                             int degree,
                             double gamma,
                             double coef0,
                             int n_support,
                             double b,
                             double* dual_coefs,
                             double* x_support,
                             int n_classes,
                             double* unique_labels,
                             double* preds,
                             double buffer_size,
                             int predict_class);
/** @} */
#ifdef __cplusplus
}
#endif
