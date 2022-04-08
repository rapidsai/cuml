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

#include <cuml/tsa/arima_common.h>

namespace raft {
class handle_t;
}

namespace ML {

/**
 * An ARIMA specialized batched kalman filter to evaluate ARMA parameters and
 * provide the resulting prediction as well as loglikelihood fit.
 *
 * @param[in]  handle        cuML handle
 * @param[in]  arima_mem     Pre-allocated temporary memory
 * @param[in]  d_ys          Batched time series
 *                           Shape (nobs, batch_size) (col-major, device)
 * @param[in]  d_exog        Batched exogenous variables
 *                           Shape (nobs, n_exog * batch_size) (col-major, device)
 * @param[in]  nobs          Number of samples per time series
 * @param[in]  params        ARIMA parameters (device)
 * @param[in]  order         ARIMA hyper-parameters
 * @param[in]  batch_size    Number of series making up the batch
 * @param[out] d_loglike     Resulting log-likelihood (per series) (device)
 * @param[out] d_pred        Predictions
 *                           shape=(nobs-d-s*D, batch_size) (device)
 * @param[in]  fc_steps      Number of steps to forecast
 * @param[in]  d_fc          Array to store the forecast
 * @param[in]  d_exog_fut    Future values of exogenous variables
 *                           Shape (fc_steps, n_exog * batch_size) (col-major, device)
 * @param[in]  level         Confidence level for prediction intervals. 0 to
 *                           skip the computation. Else 0 < level < 1
 * @param[out] d_lower       Lower limit of the prediction interval
 * @param[out] d_upper       Upper limit of the prediction interval
 */
void batched_kalman_filter(raft::handle_t& handle,
                           const ARIMAMemory<double>& arima_mem,
                           const double* d_ys,
                           const double* d_exog,
                           int nobs,
                           const ARIMAParams<double>& params,
                           const ARIMAOrder& order,
                           int batch_size,
                           double* d_loglike,
                           double* d_pred,
                           int fc_steps             = 0,
                           double* d_fc             = nullptr,
                           const double* d_exog_fut = nullptr,
                           double level             = 0,
                           double* d_lower          = nullptr,
                           double* d_upper          = nullptr);

/**
 * Convenience function for batched "jones transform" used in ARIMA to ensure
 * certain properties of the AR and MA parameters (takes host array and
 * returns host array)
 *
 * @param[in]  handle     cuML handle
 * @param[in]  arima_mem  Pre-allocated temporary memory
 * @param[in]  order      ARIMA hyper-parameters
 * @param[in]  batch_size Number of time series analyzed.
 * @param[in]  isInv      Do the inverse transform?
 * @param[in]  h_params   ARIMA parameters by batch (mu, ar, ma) (host)
 * @param[out] h_Tparams  Transformed ARIMA parameters
 *                        (expects pre-allocated array of size
 *                         (p+q)*batch_size) (host)
 */
void batched_jones_transform(raft::handle_t& handle,
                             const ARIMAMemory<double>& arima_mem,
                             const ARIMAOrder& order,
                             int batch_size,
                             bool isInv,
                             const double* h_params,
                             double* h_Tparams);
}  // namespace ML
