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
namespace Explainer{

/**
 * Generates samples of dataset for kernel shap algorithm.
 *
 *
 * @param[in]    handle             cuML handle
 * @param[inout] X                  generated data [on device] 1-0
 * @param[in]    nrows_X            number of rows in X
 * @param[in]    M                  number of columns in X
 * @param[in]    background         background data [on device]
 * @param[in]    nrows_background   number of rows in backround dataset
 * @param[out]   combinations       generated data [on device] observation=background
 * @param[in]    observation        row to scatter
 * @param[in]    nsamples           vector with number of entries that are randomly sampled
 * @param[in]    len_nsamples       number of entries to be sampled
 * @param[in]    maxsample          size of the biggest sampled observation
 * @param[in]    seed               Seed for the random number generator
 * @{
 */
void kernel_dataset(const raft::handle_t& handle, int* X, int nrows_X,
                    int M, float* background, int nrows_background,
                    float* combinations, float* observation,
                    int* nsamples, int len_nsamples, int maxsample,
                    uint64_t seed = 0ULL);

void kernel_dataset(const raft::handle_t& handle, int* X, int nrows_X,
                    int M, double* background, int nrows_background,
                    double* combinations, double* observation,
                    int* nsamples, int len_nsamples, int maxsample,
                    uint64_t seed = 0ULL);

}  // namespace Datasets
}  // namespace ML
