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

/**
 * An ARIMA specialized batched kalman filter to evaluate ARMA parameters and
 * provide the resulting prediction as well as loglikelihood fit.
 *
 * @param[in]  handle          cuML handle
 * @param[in]  d_ys_b          Batched time series
 *                             Shape (nobs, batch_size) (col-major, device)
 * @param[in]  nobs            Number of samples per time series
 * @param[in]  d_ar            AR parameters, in groups of size `p`
 *                             with total length `p * batch_size` (device)
 * @param[in]  d_ma            MA parameters, in groups of size `q`
 *                             with total length `q * batch_size` (device)
 * @param[in]  d_sar           Seasonal AR parameters, in groups of size `P`
 *                             with total length `P * batch_size` (device)
 * @param[in]  d_sma           Seasonal MA parameters, in groups of size `Q`
 *                             with total length `Q * batch_size` (device)
 * @param[in]  d_sigma2        Variance parameter. Shape: (batch_size,) (device)
 * @param[in]  p               Number of AR parameters
 * @param[in]  q               Number of MA parameters
 * @param[in]  P               Number of seasonal AR parameters
 * @param[in]  Q               Number of seasonal MA parameters
 * @param[in]  s               Seasonal period
 * @param[in]  batch_size      Number of series making up the batch
 * @param[out] loglike_b       Resulting loglikelihood (for each series)
 * @param[out] d_vs            Residual between the prediction and the
 *                             original series.
 *                             shape=(nobs, batch_size) (device)
 * @param[in]  host_loglike    Whether loglike is a host pointer
 * @param[in]  initP_kalman_it Initialize the Kalman filter covariance `P`
 *                             with 1 or more kalman iterations instead of
 *                             an analytical heuristic.
 * @param[in]  fc_steps        Number of steps to forecast
 * @param[in]  d_fc            Array to store the forecast
 */
void batched_kalman_filter(cumlHandle& handle, const double* d_ys_b, int nobs,
                           const double* d_ar, const double* d_ma,
                           const double* d_sar, const double* d_sma,
                           const double* d_sigma2, int p, int q, int P, int Q,
                           int s, int batch_size, double* loglike, double* d_vs,
                           bool host_loglike = true,
                           bool initP_kalman_it = false, int fc_steps = 0,
                           double* d_fc = nullptr);

/**
 * Public interface to batched "jones transform" used in ARIMA to ensure
 * certain properties of the AR and MA parameters.
 *
 * @param[in]  handle     cuML handle
 * @param[in]  p          Number of AR parameters
 * @param[in]  q          Number of MA parameters
 * @param[in]  P          Number of seasonal AR parameters
 * @param[in]  Q          Number of seasonal MA parameters
 * @param[in]  batch_size Number of time series analyzed.
 * @param[in]  isInv      Do the inverse transform?
 * @param[in]  d_ar       AR parameters (device)
 * @param[in]  d_ma       MA parameters (device)
 * @param[in]  d_sar      Seasonal AR parameters (device)
 * @param[in]  d_sma      Seasonal MA parameters (device)
 * @param[out] d_Tar      Transformed AR parameters (device)
 *                        Allocated internally.
 * @param[out] d_Tma      Transformed MA parameters (device)
 *                        Allocated internally.
 * @param[out] d_Tsar     Transformed seasonal AR parameters (device)
 *                        Allocated internally.
 * @param[out] d_Tsma     Transformed seasonal MA parameters (device)
 *                        Allocated internally.
 */
void batched_jones_transform(cumlHandle& handle, int p, int q, int P, int Q,
                             int batch_size, bool isInv, const double* d_ar,
                             const double* d_ma, const double* d_sar,
                             const double* d_sma, double* d_Tar, double* d_Tma,
                             double* d_Tsar, double* d_Tsma);

/**
 * Convenience function for batched "jones transform" used in ARIMA to ensure
 * certain properties of the AR and MA parameters. (takes host array and
 * returns host array)
 *
 * @param[in]  handle     cuML handle
 * @param[in]  p          Number of AR parameters
 * @param[in]  q          Number of MA parameters
 * @param[in]  P          Number of seasonal AR parameters
 * @param[in]  Q          Number of seasonal MA parameters
 * @param[in]  intercept  Whether the model fits an intercept
 * @param[in]  batch_size Number of time series analyzed.
 * @param[in]  isInv      Do the inverse transform?
 * @param[in]  h_params   Linearized ARIMA parameters by batch (mu, ar, ma) (host)
 * @param[out] h_Tparams  Transformed ARIMA parameters
 *                        (expects pre-allocated array of size (p+q)*batch_size) (host)
 */
void batched_jones_transform(cumlHandle& handle, int p, int q, int P, int Q,
                             int intercept, int batch_size, bool isInv,
                             const double* h_params, double* h_Tparams);
}  // namespace ML
