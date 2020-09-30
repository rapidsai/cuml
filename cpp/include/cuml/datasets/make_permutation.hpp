/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuml/cuml.hpp>

namespace ML {
namespace Datasets {

/**
 * Generates a dataset by tiling the `background` matrix into `out`, while
 *  adding a forward and backward permutation pass of the observation `row`
 * on the positions defined by `idx`. Example:
 *
 * background = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
 * idx = [2, 0, 3]
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
 * @param[out] out             generated data [on device] [dim = (2 * n_cols * n_rows + n_rows) * n_cols]
 * @param[in] background       background data [on device] [dim = n_cols * n_rows]
 * @param[in] n_rows           number of rows in background dataset
 * @param[in] n_cols           number of columns
 * @param[in] row              row to scatter in a permutated fashion [dim = n_cols]
 * @param[in] idx              permutation indexes [dim = n_cols]
 * @param[in]
 * @{
 */
void make_permutation(const raft::handle_t& handle, float* out,
                      float* background, int n_rows, int n_cols,
                      float* row, int* idx, bool rowMajor);

void make_permutation(const raft::handle_t& handle, double* out,
                      double* background, int n_rows, int n_cols,
                      double* row, int* idx, bool rowMajor);

/**
 * Generates a dataset by tiling the `background` matrix into `out`, while
 *  adding a forward and backward permutation pass of the observation `row`
 * on the positions defined by `idx`. Example:
 *
 * background = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
 * idx = [2, 0, 3]
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
 * @param[out] out             generated data [on device] [dim = (2 * n_cols * n_rows + n_rows) * n_cols]
 * @param[in] background       background data [on device] [dim = n_cols * n_rows]
 * @param[in] n_rows           number of rows in background dataset
 * @param[in] n_cols           number of columns
 * @param[in] row              row to scatter in a permutated fashion [dim = n_cols]
 * @param[in] idx              permutation indexes [dim = n_cols]
 * @param[in]
 * @{
 */

void single_entry_scatter(const raft::handle_t& handle, float* out,
                          float* background, int n_rows, int n_cols,
                          float* row, int* idx, bool rowMajor);

void single_entry_scatter(const raft::handle_t& handle, double* out,
                          double* background, int n_rows, int n_cols,
                          double* row, int* idx, bool rowMajor);

}  // namespace Datasets
}  // namespace ML
