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

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Flat C API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param[in] handle the cuml handle to use
 * @param[in] input an array of pointers to the input arrays
 * @param[in] size an array of sizes of input arrays
 * @param[in] n_params array size of input and sizes
 * @param[in] D the dimensionality of the arrays
 * @param[in] search_items array of items to search of dimensionality D
 * @param[in] n number of rows in search_items
 * @param[out] res_I the resulting index array of size n * k
 * @param[out] res_D the resulting distance array of size n * k
 * @param[in] k the number of nearest neighbors to return
 * @param[in] rowMajorIndex is the index array in row major layout?
 * @param[in] rowMajorQuery is the query array in row major layout?
 * @param[in] metric_type the type of distance metric to use. This corresponds
 *            to the value in the cuml::ML::distance::DistanceType enum.
 *            Default is Euclidean (L2).
 * @param[in] metric_arg the value of `p` for Minkowski (l-p) distances. This
 *            is ignored if the metric_type is not Minkowski.
 * @param[in] expanded should lp-based distances be returned in their expanded
 *            form (e.g., without raising to the 1/p power).
 */
cumlError_t knn_search(const cumlHandle_t handle,
                       float** input,
                       int* size,
                       int n_params,
                       int D,
                       float* search_items,
                       int n,
                       int64_t* res_I,
                       float* res_D,
                       int k,
                       bool rowMajorIndex,
                       bool rowMajorQuery,
                       int metric_type,
                       float metric_arg,
                       bool expanded);

#ifdef __cplusplus
}
#endif
