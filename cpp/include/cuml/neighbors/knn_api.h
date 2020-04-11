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

#include <cuml_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Flat C API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param handle the cuml handle to use
 * @param input an array of pointers to the input arrays
 * @param size an array of sizes of input arrays
 * @param n_params array size of input and sizes
 * @param D the dimensionality of the arrays
 * @param search_items array of items to search of dimensionality D
 * @param n number of rows in search_items
 * @param res_I the resulting index array of size n * k
 * @param res_D the resulting distance array of size n * k
 * @param k the number of nearest neighbors to return
 * @param rowMajorIndex is the index array in row major layout?
 * @param rowMajorQuery is the query array in row major layout?
 */
cumlError_t knn_search(const cumlHandle_t handle, float **input, int *size,
                       int n_params, int D, const float *search_items, int n,
                       int64_t *res_I, float *res_D, int k, bool rowMajorIndex,
                       bool rowMajorQuery);

#ifdef __cplusplus
}
#endif
