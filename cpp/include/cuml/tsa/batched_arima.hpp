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

enum LoglikeMethod { CSS, MLE };

/**
 * Pack separate parameter arrays into a compact array
 *
 * @param[in]  handle     cuML handle
 * @param[in]  params     Parameter structure
 * @param[in]  order      ARIMA order
 * @param[in]  batch_size Batch size
 * @param[out] param_vec  Compact parameter array
 */
void pack(raft::handle_t& handle,
          const ARIMAParams<double>& params,
          const ARIMAOrder& order,
          int batch_size,
          double* param_vec);

/**
 * Unpack a compact array into separate parameter arrays
 *
 * @param[in]  handle     cuML handle
 * @param[out] params     Parameter structure
 * @param[in]  order      ARIMA order
 * @param[in]  batch_size Batch size
 * @param[in]  param_vec  Compact parameter array
 */
void unpack(raft::handle_t& handle,
            ARIMAParams<double>& params,
            const ARIMAOrder& order,
            int batch_size,
            const double* param_vec);

/**
 * Detect missing observations in a time series
 *
 * @param[in]  handle     cuML handle
 * @param[in]  d_y        Time series
 * @param[in]  n_elem     Total number of elements in the dataset
 */
bool detect_missing(raft::handle_t& handle, const double* d_y, int n_elem);

/**
 * Compute the differenced series (seasonal and/or non-seasonal differences)
 *
 * @param[in]  handle     cuML handle
 * @param[out] d_y_diff   Differenced series
 * @param[in]  d_y        Original series
 * @param[in]  batch_size Batch size
 * @param[in]  n_obs      Number of observations
 * @param[in]  order      ARIMA order
 */
void batched_diff(raft::handle_t& handle,
                  double* d_y_diff,
                  const double* d_y,
                  int batch_size,
                  int n_obs,
                  const ARIMAOrder& order);

/**
 * Compute the loglikelihood of the given parameter on the given time series
 * in a batched context.
 *
 * @param[in]  handle       cuML handle
 * @param[in]  arima_mem    Pre-allocated temporary memory
 * @param[in]  d_y          Series to fit: shape = (n_obs, batch_size) and
 *                          expects column major data layout. (device)
 * @param[in]  d_exog       Exogenous variables: shape = (n_obs, n_exog * batch_size) and
 *                          expects column major data layout. (device)
 * @param[in]  batch_size   Number of time series
 * @param[in]  n_obs        Number of observations in a time series
 * @param[in]  order        ARIMA hyper-parameters
 * @param[in]  d_params     Parameters to evaluate grouped by series:
 *                          [mu0, ar.., ma.., mu1, ..] (device)
 * @param[out] loglike      Log-Likelihood of the model per series
 * @param[in]  trans        Run `jones_transform` on params.
 * @param[in]  host_loglike Whether loglike is a host pointer
 * @param[in]  method       Whether to use sum-of-squares or Kalman filter
 * @param[in]  truncate     For CSS, start the sum-of-squares after a given
 *                          number of observations
 */
void batched_loglike(raft::handle_t& handle,
                     const ARIMAMemory<double>& arima_mem,
                     const double* d_y,
                     const double* d_exog,
                     int batch_size,
                     int n_obs,
                     const ARIMAOrder& order,
                     const double* d_params,
                     double* loglike,
                     bool trans           = true,
                     bool host_loglike    = true,
                     LoglikeMethod method = MLE,
                     int truncate         = 0);

/**
 * Compute the loglikelihood of the given parameter on the given time series
 * in a batched context.
 *
 * @note: this overload should be used when the parameters are already unpacked
 *        to avoid useless packing / unpacking
 *
 * @param[in]  handle       cuML handle
 * @param[in]  arima_mem    Pre-allocated temporary memory
 * @param[in]  d_y          Series to fit: shape = (n_obs, batch_size) and
 *                          expects column major data layout. (device)
 * @param[in]  d_exog       Exogenous variables: shape = (n_obs, n_exog * batch_size) and
 *                          expects column major data layout. (device)
 * @param[in]  batch_size   Number of time series
 * @param[in]  n_obs        Number of observations in a time series
 * @param[in]  order        ARIMA hyper-parameters
 * @param[in]  params       ARIMA parameters (device)
 * @param[out] loglike      Log-Likelihood of the model per series
 * @param[in]  trans        Run `jones_transform` on params.
 * @param[in]  host_loglike Whether loglike is a host pointer
 * @param[in]  method       Whether to use sum-of-squares or Kalman filter
 * @param[in]  truncate     For CSS, start the sum-of-squares after a given
 *                          number of observations
 * @param[in]  fc_steps     Number of steps to forecast
 * @param[in]  d_fc         Array to store the forecast
 * @param[in]  d_exog_fut   Future values of exogenous variables
 *                          Shape (fc_steps, n_exog * batch_size) (col-major, device)
 * @param[in]  level        Confidence level for prediction intervals. 0 to
 *                          skip the computation. Else 0 < level < 1
 * @param[out] d_lower      Lower limit of the prediction interval
 * @param[out] d_upper      Upper limit of the prediction interval
 */
void batched_loglike(raft::handle_t& handle,
                     const ARIMAMemory<double>& arima_mem,
                     const double* d_y,
                     const double* d_exog,
                     int batch_size,
                     int n_obs,
                     const ARIMAOrder& order,
                     const ARIMAParams<double>& params,
                     double* loglike,
                     bool trans               = true,
                     bool host_loglike        = true,
                     LoglikeMethod method     = MLE,
                     int truncate             = 0,
                     int fc_steps             = 0,
                     double* d_fc             = nullptr,
                     const double* d_exog_fut = nullptr,
                     double level             = 0,
                     double* d_lower          = nullptr,
                     double* d_upper          = nullptr);

/**
 * Compute the gradient of the log-likelihood
 *
 * @param[in]  handle       cuML handle
 * @param[in]  arima_mem    Pre-allocated temporary memory
 * @param[in]  d_y          Series to fit: shape = (n_obs, batch_size) and
 *                          expects column major data layout. (device)
 * @param[in]  d_exog       Exogenous variables: shape = (n_obs, n_exog * batch_size) and
 *                          expects column major data layout. (device)
 * @param[in]  batch_size   Number of time series
 * @param[in]  n_obs        Number of observations in a time series
 * @param[in]  order        ARIMA hyper-parameters
 * @param[in]  d_x          Parameters grouped by series
 * @param[out] d_grad       Gradient to compute
 * @param[in]  h            Finite-differencing step size
 * @param[in]  trans        Run `jones_transform` on params
 * @param[in]  method       Whether to use sum-of-squares or Kalman filter
 * @param[in]  truncate     For CSS, start the sum-of-squares after a given
 *                          number of observations
 */
void batched_loglike_grad(raft::handle_t& handle,
                          const ARIMAMemory<double>& arima_mem,
                          const double* d_y,
                          const double* d_exog,
                          int batch_size,
                          int n_obs,
                          const ARIMAOrder& order,
                          const double* d_x,
                          double* d_grad,
                          double h,
                          bool trans           = true,
                          LoglikeMethod method = MLE,
                          int truncate         = 0);

/**
 * Batched in-sample and out-of-sample prediction of a time-series given all
 * the model parameters
 *
 * @param[in]  handle      cuML handle
 * @param[in]  arima_mem   Pre-allocated temporary memory
 * @param[in]  d_y         Batched Time series to predict.
 *                         Shape: (num_samples, batch size) (device)
 * @param[in]  d_exog      Exogenous variables.
 *                         Shape = (n_obs, n_exog * batch_size) (device)
 * @param[in]  d_exog_fut  Future values of exogenous variables
 *                         Shape: (end - n_obs, batch_size) (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  n_obs       Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  start       Index to start the prediction
 * @param[in]  end         Index to end the prediction (excluded)
 * @param[in]  order       ARIMA hyper-parameters
 * @param[in]  params      ARIMA parameters (device)
 * @param[out] d_y_p       Prediction output (device)
 * @param[in]  pre_diff    Whether to use pre-differencing
 * @param[in]  level       Confidence level for prediction intervals. 0 to
 *                         skip the computation. Else 0 < level < 1
 * @param[out] d_lower     Lower limit of the prediction interval
 * @param[out] d_upper     Upper limit of the prediction interval
 */
void predict(raft::handle_t& handle,
             const ARIMAMemory<double>& arima_mem,
             const double* d_y,
             const double* d_exog,
             const double* d_exog_fut,
             int batch_size,
             int n_obs,
             int start,
             int end,
             const ARIMAOrder& order,
             const ARIMAParams<double>& params,
             double* d_y_p,
             bool pre_diff   = true,
             double level    = 0,
             double* d_lower = nullptr,
             double* d_upper = nullptr);

/**
 * Compute an information criterion (AIC, AICc, BIC)
 *
 * @param[in]  handle      cuML handle
 * @param[in]  arima_mem   Pre-allocated temporary memory
 * @param[in]  d_y         Series to fit: shape = (n_obs, batch_size) and
 *                         expects column major data layout. (device)
 * @param[in]  d_exog      Exogenous variables.
 *                         Shape = (n_obs, n_exog * batch_size) (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  n_obs       Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  order       ARIMA hyper-parameters
 * @param[in]  params      ARIMA parameters (device)
 * @param[out] ic          Array where to write the information criteria
 *                         Shape: (batch_size) (device)
 * @param[in]  ic_type     Type of information criterion wanted.
 *                         0: AIC, 1: AICc, 2: BIC
 */
void information_criterion(raft::handle_t& handle,
                           const ARIMAMemory<double>& arima_mem,
                           const double* d_y,
                           const double* d_exog,
                           int batch_size,
                           int n_obs,
                           const ARIMAOrder& order,
                           const ARIMAParams<double>& params,
                           double* ic,
                           int ic_type);

/**
 * Provide initial estimates to ARIMA parameters mu, AR, and MA
 *
 * @param[in]  handle      cuML handle
 * @param[in]  params      ARIMA parameters (device)
 * @param[in]  d_y         Series to fit: shape = (n_obs, batch_size) and
 *                         expects column major data layout. (device)
 * @param[in]  d_exog      Exogenous variables.
 *                         Shape = (n_obs, n_exog * batch_size) (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  n_obs       Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  order       ARIMA hyper-parameters
 * @param[in]  missing     Are there missing observations?
 */
void estimate_x0(raft::handle_t& handle,
                 ARIMAParams<double>& params,
                 const double* d_y,
                 const double* d_exog,
                 int batch_size,
                 int n_obs,
                 const ARIMAOrder& order,
                 bool missing);

}  // namespace ML
