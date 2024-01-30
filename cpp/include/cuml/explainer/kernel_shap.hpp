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

#include <cstdint>

namespace raft {
class handle_t;
}

namespace ML {
namespace Explainer {

/**
 * Generates samples of dataset for kernel shap algorithm.
 *
 *
 * @param[in]    handle             cuML handle
 * @param[inout] X                  generated data [on device] 1-0 (row major)
 * @param[in]    nrows_X            number of rows in X
 * @param[in]    ncols              number of columns in X, background and dataset
 * @param[in]    background         background data [on device]
 * @param[in]    nrows_background   number of rows in background dataset
 * @param[out]   dataset            generated data [on device] observation=background (row major)
 * @param[in]    observation        row to scatter
 * @param[in]    nsamples           vector with number of entries that are randomly sampled
 * @param[in]    len_nsamples       number of entries to be sampled
 * @param[in]    maxsample          size of the biggest sampled observation
 * @param[in]    seed               Seed for the random number generator
 *
 * Kernel distrubutes exact part of the kernel shap dataset
 * Each block scatters the data of a row of `observations` into the (number of rows of
 * background) in `dataset`, based on the row of `X`.
 * So, given:
 * background = [[0, 1, 2],
                 [3, 4, 5]]
 * observation = [100, 101, 102]
 * X = [[1, 0, 1],
 *      [0, 1, 1]]
 *
 * dataset (output):
 * [[100, 1, 102],
 *  [100, 4, 102]
 *  [0, 101, 102],
 *  [3, 101, 102]]
 * The first thread of each block calculates the sampling of `k` entries of `observation`
 * to scatter into `dataset`. Afterwards each block scatters the data of a row of `X` into
 * the (number of rows of background) in `dataset`.
 * So, given:
 * background = [[0, 1, 2, 3],
 *               [5, 6, 7, 8]]
 * observation = [100, 101, 102, 103]
 * nsamples = [3, 2]
 *
 * X (output)
 *      [[1, 0, 1, 1],
 *       [0, 1, 1, 0]]
 *
 * dataset (output):
 * [[100, 1, 102, 103],
 *  [100, 6, 102, 103]
 *  [0, 101, 102, 3],
 *  [5, 101, 102, 8]]
 */
void kernel_dataset(const raft::handle_t& handle,
                    float* X,
                    int nrows_X,
                    int ncols,
                    float* background,
                    int nrows_background,
                    float* dataset,
                    float* observation,
                    int* nsamples,
                    int len_nsamples,
                    int maxsample,
                    uint64_t seed = 0ULL);

void kernel_dataset(const raft::handle_t& handle,
                    float* X,
                    int nrows_X,
                    int ncols,
                    double* background,
                    int nrows_background,
                    double* dataset,
                    double* observation,
                    int* nsamples,
                    int len_nsamples,
                    int maxsample,
                    uint64_t seed = 0ULL);

}  // namespace Explainer
}  // namespace ML
