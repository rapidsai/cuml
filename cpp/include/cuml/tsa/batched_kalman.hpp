/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuml/tsa/arima_common.h>
#include <cuml/cuml.hpp>

namespace ML {

/**
 * An ARIMA specialized batched kalman filter to evaluate ARMA parameters and
 * provide the resulting prediction as well as loglikelihood fit.
 *
 * @param[in]  handle        cuML handle
 * @param[in]  d_ys_b        Batched time series
 *                           Shape (nobs, batch_size) (col-major, device)
 * @param[in]  nobs          Number of samples per time series
 * @param[in]  params        ARIMA parameters (device)
 * @param[in]  order         ARIMA hyper-parameters
 * @param[in]  batch_size    Number of series making up the batch
 * @param[out] loglike_b     Resulting loglikelihood (for each series)
 * @param[out] d_vs          Residual between the prediction and the
 *                           original series.
 *                           shape=(nobs-d-s*D, batch_size) (device)
 * @param[in]  host_loglike  Whether loglike is a host pointer
 * @param[in]  fc_steps      Number of steps to forecast
 * @param[in]  d_fc          Array to store the forecast
 */
void batched_kalman_filter(cumlHandle& handle, const double* d_ys_b, int nobs,
                           const ARIMAParams<double>& params,
                           const ARIMAOrder& order, int batch_size,
                           double* loglike, double* d_vs,
                           bool host_loglike = true, int fc_steps = 0,
                           double* d_fc = nullptr);

/**
 * Convenience function for batched "jones transform" used in ARIMA to ensure
 * certain properties of the AR and MA parameters (takes host array and
 * returns host array)
 *
 * @param[in]  handle     cuML handle
 * @param[in]  order      ARIMA hyper-parameters
 * @param[in]  batch_size Number of time series analyzed.
 * @param[in]  isInv      Do the inverse transform?
 * @param[in]  h_params   ARIMA parameters by batch (mu, ar, ma) (host)
 * @param[out] h_Tparams  Transformed ARIMA parameters
 *                        (expects pre-allocated array of size
 *                         (p+q)*batch_size) (host)
 */
void batched_jones_transform(cumlHandle& handle, const ARIMAOrder& order,
                             int batch_size, bool isInv, const double* h_params,
                             double* h_Tparams);
}  // namespace ML
