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

#ifndef _cplusplus
extern "C" {
#endif

/**
 * Fit unsupervised UMAP
 * @param[in] handle
 *            cuml handle to use across the algorithm
 * @param[in] X
 *            device pointer to input matrix (num_samples * num_dims in row-major order)
 * @param[in] num_samples
 *            number of rows in matrix X
 * @param[in] num_dims
 *            number of columns in matrix X
 * @param[out] embeddings
 *             device pointer that holds the embeddings (num_samples * n_components, row-major order)
 * @param[in] n_components
 *            number of features in the final embedding
 * @return CUML_SUCCESS on success and other corresponding flags upon any failures.
 */
cumlError_t cumlSpUmapFit(cumlHandle_t handle, float *X, int num_samples,
                          int num_dims, float *embeddings, int n_components);
/** @} */

/**
 * Fit supervised UMAP
 * @param[in] handle
 *            cuml handle to use across the algorithm
 * @param[in] X
 *            device pointer to input matrix (num_samples * num_dims in row-major order)
 * @param[in] y
 *            device pointer to label vector (num_samples length)
 * @param[in] num_samples
 *            number of rows in matrix X
 * @param[in] num_dims
 *            number of columns in matrix X
 * @param[out] embeddings
 *             device pointer that holds the embeddings (num_samples * n_components, row-major order)
 * @param[in] n_components
 *            number of features in the final embedding
 * @return CUML_SUCCESS on success and other corresponding flags upon any failures.
 */
cumlError_t cumlSpUmapFitSupervised(cumlHandle_t handle, float *X, float *y,
                                    int num_samples, int num_dims,
                                    float *embeddings, int n_components);

/** @} */

#ifdef __cplusplus
}
#endif
