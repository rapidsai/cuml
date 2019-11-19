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
#include <cuml/cuml.hpp>

namespace ML {

namespace Stationarity {

/**
 * @brief Compute recommended trend parameter (d=0 or 1) for a batched series
 * 
 * @details This function operates a stationarity test on the given series
 *          and for the series that fails the test, differentiates them
 *          and runs the test again on the first difference.
 * 
 * @note The data is a column-major matrix where the series are columns.
 *       The output is an array of size n_batches.
 * 
 * @param[in]   handle          cuML handle
 * @param[in]   y_d             Input data
 * @param[out]  d               Integer array to store the trends
 * @param[in]   n_batches       Number of batches
 * @param[in]   n_samples       Number of samples
 * @param[in]   pval_threshold  P-value threshold above which a series is
 *                              considered stationary
 * 
 * @return      An integer to track if some series failed the test
 * @retval  -1  Some series failed the test
 * @retval   0  All series passed the test for d=0
 * @retval   1  Some series passed for d=0, the others for d=1
 */
int stationarity(const cumlHandle& handle, const float* y_d, int* d,
                 int n_batches, int n_samples, float pval_threshold);
int stationarity(const cumlHandle& handle, const double* y_d, int* d,
                 int n_batches, int n_samples, double pval_threshold);

}  // namespace Stationarity
}  // namespace ML
