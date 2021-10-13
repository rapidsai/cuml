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

namespace raft {
class handle_t;
}

namespace ML {

namespace Stationarity {

/**
 * @brief Perform the KPSS stationarity test on the data differenced according
 *        to the given order
 *
 * @param[in]   handle          cuML handle
 * @param[in]   d_y             Input data (column-major, series in columns)
 * @param[out]  results         Boolean device array to store the results
 * @param[in]   batch_size      Batch size
 * @param[in]   n_obs           Number of observations
 * @param[in]   d               Order of simple differencing
 * @param[out]  D               Order of seasonal differencing
 * @param[in]   s               Seasonal period if D > 0 (else unused)
 * @param[in]   pval_threshold  P-value threshold above which a series is
 *                              considered stationary
 */
void kpss_test(const raft::handle_t& handle,
               const float* d_y,
               bool* results,
               int batch_size,
               int n_obs,
               int d,
               int D,
               int s,
               float pval_threshold);
void kpss_test(const raft::handle_t& handle,
               const double* d_y,
               bool* results,
               int batch_size,
               int n_obs,
               int d,
               int D,
               int s,
               double pval_threshold);

}  // namespace Stationarity
}  // namespace ML
