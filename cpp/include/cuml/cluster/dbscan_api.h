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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DbscanC C-wrapper to C++ implementation of Dbscan algo
 * @brief Fits a DBSCAN model on an input feature matrix and outputs the labels.
 * @param[in] handle cuml handle to use across the algorithm
 * @param[in] input row-major input feature matrix
 * @param[in] n_rows number of samples in the input feature matrix
 * @param[in] n_cols number of features in the input feature matrix
 * @param[in] eps the epsilon value to use for epsilon-neighborhood determination
 * @param[in] min_pts minimum number of points to determine a cluster
 * @param[out] labels (size n_rows) output labels array
 * @param[out] core_sample_indices (size n_rows) output array containing the
 *             indices of each core point. If the number of core points is less than n_rows, the
 * right will be padded with -1. Setting this to NULL will prevent calculating the core sample
 * indices
 * @param[in] max_mem_bytes the maximum number of bytes to be used for each batch of
 *            the pairwise distance calculation. This enables the trade off between
 *            memory usage and algorithm execution time.
 * @param[in] verbosity Set a verbosity level (higher values means quieter)
 *                      Refer to `cuml/common/logger.hpp` for these levels
 * @return CUML_SUCCESS on success and other corresponding flags upon any failures.
 * @{
 */
cumlError_t cumlSpDbscanFit(cumlHandle_t handle,
                            float* input,
                            int n_rows,
                            int n_cols,
                            float eps,
                            int min_pts,
                            int* labels,
                            int* core_sample_indices,
                            size_t max_bytes_per_batch,
                            int verbosity);

cumlError_t cumlDpDbscanFit(cumlHandle_t handle,
                            double* input,
                            int n_rows,
                            int n_cols,
                            double eps,
                            int min_pts,
                            int* labels,
                            int* core_sample_indices,
                            size_t max_bytes_per_batch,
                            int verbosity);
/** @} */

/**
 * @defgroup DbscanC C-wrapper to C++ implementation of Dbscan algo
 * @brief Fits a DBSCAN model on an input feature matrix and outputs the labels.
 * @param[in] handle cuml handle to use across the algorithm
 * @param[in] input row-major and concatenated input feature matrixes from different groups
 * @param[in] n_groups number of groups of multiple input feature matrixes
 * @param[in] n_rows_ptr ponter to the numbers of samples in the grouped input feature matrixes
 * @param[in] n_cols number of features in the grouped input feature matrixes
 * @param[in] eps_ptr pointer to the epsilon values to use for epsilon-neighborhood determination
 * @param[in] min_pts_ptr pointer to the minimum numbers of points to determine a cluster for each
 * group
 * @param[out] labels (size sum(n_rows)) output labels array
 * @param[out] core_sample_indices (size sum(n_rows)) output array containing the
 *             indices of each core point. If the number of core points of each group is less
 * than n_rows, the right will be padded with -1. Setting this to NULL will prevent calculating
 * the core sample indices
 * @param[in] max_mem_bytes the maximum number of bytes to be used for each batch of
 *            the pairwise distance calculation. This enables the trade off between
 *            memory usage and algorithm execution time.
 * @param[in] verbosity Set a verbosity level (higher values means quieter)
 *                      Refer to `cuml/common/logger.hpp` for these levels
 * @param[in] custom_workspace workspace buffer provided by user
 * @param[in] custom_workspace_size required size of workspace buffer provided by user
 * @return CUML_SUCCESS on success and other corresponding flags upon any failures.
 * @{
 */
cumlError_t cumlMultiSpDbscanFit(cumlHandle_t handle,
                                 float* input,
                                 int n_groups,
                                 int* n_rows_ptr,
                                 int n_cols,
                                 const float* eps_ptr,
                                 const int* min_pts_ptr,
                                 int* labels,
                                 int* core_sample_indices,
                                 size_t max_bytes_per_batch,
                                 int verbosity,
                                 void* custom_workspace,
                                 size_t* custom_workspace_size);

cumlError_t cumlMultiDpDbscanFit(cumlHandle_t handle,
                                 double* input,
                                 int n_groups,
                                 int* n_rows_ptr,
                                 int n_cols,
                                 const double* eps_ptr,
                                 const int* min_pts_ptr,
                                 int* labels,
                                 int* core_sample_indices,
                                 size_t max_bytes_per_batch,
                                 int verbosity,
                                 void* custom_workspace,
                                 size_t* custom_workspace_size);
/** @} */

#ifdef __cplusplus
}
#endif
