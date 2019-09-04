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

#include <string>
#include <vector>

namespace ML {

//! An ARIMA specialized batched kalman filter to evaluate ARMA parameters and
//! provide the resulting prediction as well as loglikelihood fit.
//! @param h_ys_b The (batched) time series with shape (nobs, num_batches) in column major layout. Memory on host.
//! @param nobs The number of samples per time series
//! @param b_ar_params The AR parameters, in groups of size `p` with total length `p * num_batches`
//! @param b_ma_params The mA parameters, in groups of size `q` with total length `q * num_batches`
//! @param p The number of AR parameters
//! @param q The number of MA parameters
//! @param num_batches The number of series making up the batch
//! @param loglike_b The resulting loglikelihood (for each series)
//! @param h_vs_b The residual between the prediction and the original series.
//! @param initP_with_kalman_iterations Initialize the Kalman filter covariance `P` with 1 or more kalman iterations instead of an analytical heuristic.
void batched_kalman_filter(double* h_ys_b, int nobs,
                           const std::vector<double>& b_ar_params,
                           const std::vector<double>& b_ma_params, int p, int q,
                           int num_batches, std::vector<double>& loglike_b,
                           std::vector<std::vector<double>>& h_vs_b,
                           bool initP_with_kalman_iterations = false);

//! NVTX Wrapper function creating a range.
//! @param msg The NVTX range name.
void nvtx_range_push(std::string msg);

//! NVTX wrapper function ending a range.
void nvtx_range_pop();

//! Public interface to batched "jones transform" used in ARIMA to ensure
//! certain properties of the AR and MA parameters.
//! @param p Number of AR parameters
//! @param q Number of MA parameters
//! @param num_batches Number of time series analyzed.
//! @param isInv Do the inverse transform?
//! @param ar AR parameters (host)
//! @param ma MA parameters (host)
//! @param ar Transformed AR parameters (host)
//! @param ma Transformed MA parameters (host)
void batched_jones_transform(int p, int q, int num_batches, bool isInv,
                             const std::vector<double>& ar,
                             const std::vector<double>& ma,
                             std::vector<double>& Tar,
                             std::vector<double>& Tma);

}  // namespace ML
