/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

namespace raft {
class handle_t;
}

namespace ML {
namespace Explainer {

/**
 * Generates a dataset by tiling the `background` matrix into `out`, while
 *  adding a forward and backward permutation pass of the observation `row`
 * on the positions defined by `idx`. Example:
 *
 * background = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
 * idx = [2, 0, 1]
 * row = [100, 101, 102]
 * output:
 * [[  0,   1,   2]
 *  [  3,   4,   5]
 *  [  6,   7,   8]
 *  [  0,   1, 102]
 *  [  3,   4, 102]
 *  [  6,   7, 102]
 *  [100,   1, 102]
 *  [100,   4, 102]
 *  [100,   7, 102]
 *  [100, 101, 102]
 *  [100, 101, 102]
 *  [100, 101, 102]
 *  [100, 101,   2]
 *  [100, 101,   5]
 *  [100, 101,   8]
 *  [  0, 101,   2]
 *  [  3, 101,   5]
 *  [  6, 101,   8]
 *  [  0,   1,   2]
 *  [  3,   4,   5]
 *  [  6,   7,   8]]
 *
 *
 * @param[in]  handle          cuML handle
 * @param[out] dataset         generated data in either row major or column major
 *                             format, depending on the `row_major` parameter [on device]
 *                             [dim = (2 * ncols * nrows_bg + nrows_bg) * ncols]
 * @param[in] background       background data [on device] [dim = ncols * nrows_bg]
 * @param[in] nrows_bg         number of rows in background dataset
 * @param[in] ncols            number of columns
 * @param[in] row              row to scatter in a permutated fashion [dim = ncols]
 * @param[in] idx              permutation indexes [dim = ncols]
 * @param[in] row_major        boolean to generate either row or column major data
 *
 */
void permutation_shap_dataset(const raft::handle_t& handle,
                              float* dataset,
                              const float* background,
                              int nrows_bg,
                              int ncols,
                              const float* row,
                              int* idx,
                              bool row_major);

void permutation_shap_dataset(const raft::handle_t& handle,
                              double* dataset,
                              const double* background,
                              int nrows_bg,
                              int ncols,
                              const double* row,
                              int* idx,
                              bool row_major);

/**
 * Generates a dataset by tiling the `background` matrix into `out`, while
 *  adding a forward and backward permutation pass of the observation `row`
 * on the positions defined by `idx`. Example:
 *
 * background = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
 * idx = [2, 0, 1]
 * row = [100, 101, 102]
 * output:
 * [[  0,   1,   2]
 *  [  3,   4,   5]
 *  [  6,   7,   8]
 *  [  0,   1, 102]
 *  [  3,   4, 102]
 *  [  6,   7, 102]
 *  [100,   1,   2]
 *  [100,   4,   5]
 *  [100,   7,   8]
 *  [  0, 101,   2]
 *  [  3, 101,   5]
 *  [  6, 101,   8]]
 *
 *
 * @param[in]  handle          cuML handle
 * @param[out] dataset         generated data [on device] [dim = (2 * ncols * nrows_bg + nrows_bg) *
 * ncols]
 * @param[in] background       background data [on device] [dim = ncols * nrows_bg]
 * @param[in] nrows_bg         number of rows in background dataset
 * @param[in] ncols            number of columns
 * @param[in] row              row to scatter in a permutated fashion [dim = ncols]
 * @param[in] idx              permutation indexes [dim = ncols]
 * @param[in] row_major        boolean to generate either row or column major data
 *
 */

void shap_main_effect_dataset(const raft::handle_t& handle,
                              float* dataset,
                              const float* background,
                              int nrows_bg,
                              int ncols,
                              const float* row,
                              int* idx,
                              bool row_major);

void shap_main_effect_dataset(const raft::handle_t& handle,
                              double* dataset,
                              const double* background,
                              int nrows_bg,
                              int ncols,
                              const double* row,
                              int* idx,
                              bool row_major);

/**
 * Function that aggregates averages of the averatge of results of the model
 * called with the permutation dataset, to estimate the SHAP values.
 * It is equivalent to the Python code:
 *  for i,ind in enumerate(idx):
 *     shap_values[ind] += y_hat[i + 1] - y_hat[i]
 *  for i,ind in enumerate(idx):
 *     shap_values[ind] += y_hat[i + ncols] - y_hat[i + ncols + 1]
 *
 * @param[in]  handle          cuML handle
 * @param[out] shap_values     Array where the results are aggregated [dim = ncols]
 * @param[in] y_hat            Results to use for the aggregation [dim = ncols + 1]
 * @param[in] ncols            number of columns
 * @param[in] idx              permutation indexes [dim = ncols]
 */
void update_perm_shap_values(const raft::handle_t& handle,
                             float* shap_values,
                             const float* y_hat,
                             const int ncols,
                             const int* idx);

void update_perm_shap_values(const raft::handle_t& handle,
                             double* shap_values,
                             const double* y_hat,
                             const int ncols,
                             const int* idx);

}  // namespace Explainer
}  // namespace ML
