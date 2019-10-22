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
#include <string>
#include <vector>

namespace ML {

/* An ARIMA specialized batched kalman filter to evaluate ARMA parameters and
 * provide the resulting prediction as well as loglikelihood fit.
 * @param d_ys_b The (batched) time series with shape (nobs, num_batches) in column major layout. Memory on device.
 * @param nobs The number of samples per time series
 * @param d_b_ar_params The AR parameters, in groups of size `p` with total length `p * num_batches` (device)
 * @param d_b_ma_params The mA parameters, in groups of size `q` with total length `q * num_batches` (device)
 * @param p The number of AR parameters
 * @param q The number of MA parameters
 * @param num_batches The number of series making up the batch
 * @param loglike_b The resulting loglikelihood (for each series)
 * @param d_vs The residual between the prediction and the original series. shape=(nobs, num_batches), Memory on device.
 * @param initP_with_kalman_iterations Initialize the Kalman filter covariance `P` with 1 or more kalman iterations instead of an analytical heuristic.
 */
void batched_kalman_filter(cumlHandle& handle, double* d_ys_b, int nobs,
                           const double* d_b_ar_params,
                           const double* d_b_ma_params, int p, int q,
                           int num_batches, std::vector<double>& loglike_b,
                           double* d_vs,
                           bool initP_with_kalman_iterations = false);

/* Turns linear array of parameters into arrays of mu, ar, and ma parameters. (using device arrays)
 * @param d_params Linear array of all parameters grouped by batch [mu, ar, ma] (device)
 * @param d_mu trend parameter (device)
 * @param d_ar AR parameters (device)
 * @param d_ma MA parameters (device)
 * @param batchSize Number of time series analyzed.
 * @param p Number of AR parameters
 * @param d Trend parameter
 * @param q Number of MA parameters
 * @param stream CUDA stream
 */
void unpack(const double* d_params, double* d_mu, double* d_ar, double* d_ma,
            int batchSize, int p, int d, int q, cudaStream_t stream);

/* Public interface to batched "jones transform" used in ARIMA to ensure
 * certain properties of the AR and MA parameters.
 * @param p Number of AR parameters
 * @param d Trend parameter
 * @param q Number of MA parameters
 * @param batchSize Number of time series analyzed.
 * @param isInv Do the inverse transform?
 * @param d_ar AR parameters (device)
 * @param d_ma MA parameters (device)
 * @param d_Tar Transformed AR parameters. Allocated internally (device)
 * @param d_Tma Transformed MA parameters. Allocated internally (device)
 */
void batched_jones_transform(cumlHandle& handle, int p, int q, int batchSize,
                             bool isInv, const double* d_ar, const double* d_ma,
                             double* d_Tar, double* d_Tma);

/* Convenience function for batched "jones transform" used in ARIMA to ensure
 * certain properties of the AR and MA parameters. (takes host array and returns host array)
 * @param p Number of AR parameters
 * @param d Trend parameter
 * @param q Number of MA parameters
 * @param batchSize Number of time series analyzed.
 * @param isInv Do the inverse transform?
 * @param h_params Linearized ARIMA parameters by batch (mu, ar, ma) (host)
 * @param h_Tparams Transformed ARIMA parameters (expects pre-allocated array of size (p+q)*batchSize) (host)
 */
void batched_jones_transform(cumlHandle& handle, int p, int d, int q,
                             int batchSize, bool isInv, const double* h_params,
                             double* h_Tparams);
}  // namespace ML
