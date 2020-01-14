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
#include <vector>

namespace ML {

/**
 * Compute the loglikelihood of the given parameter on the given time series
 * in a batched context.
 *
 * @param[in]  handle       cuML handle
 * @param[in]  d_y          Series to fit: shape = (nobs, batch_size) and
 *                          expects column major data layout. (device)
 * @param[in]  batch_size   Number of time series
 * @param[in]  nobs         Number of observations in a time series
 * @param[in]  p            Number of AR parameters
 * @param[in]  d            Difference order
 * @param[in]  q            Number of MA parameters
 * @param[in]  P            Number of seasonal AR parameters
 * @param[in]  D            Seasonal difference order
 * @param[in]  Q            Number of seasonal MA parameters
 * @param[in]  s            Seasonal period
 * @param[in]  intercept    Whether the model fits an intercept (constant term)
 * @param[in]  d_params     Parameters to evaluate group by series:
 *                          [mu0, ar.., ma.., mu1, ..] (device)
 * @param[out] loglike      Log-Likelihood of the model per series (host)
 * @param[out] d_vs         The residual between model and original signal.
 *                          shape = (nobs, batch_size) (device)
 * @param[in]  trans        Run `jones_transform` on params.
 * @param[in]  host_loglike Whether loglike is a host pointer
 * @param[in]  fc_steps     Number of steps to forecast
 * @param[in]  d_fc         Array to store the forecast
 */
void batched_loglike(cumlHandle& handle, const double* d_y, int batch_size,
                     int nobs, int p, int d, int q, int P, int D, int Q, int s,
                     int intercept, const double* d_params, double* loglike,
                     double* d_vs, bool trans = true, bool host_loglike = true,
                     int fc_steps = 0, double* d_fc = nullptr);

/**
 * Compute the loglikelihood of the given parameter on the given time series
 * in a batched context.
 * 
 * @note: this overload should be used when the parameters are already unpacked
 *        to avoid useless packing / unpacking
 *
 * @param[in]  handle       cuML handle
 * @param[in]  d_y          Series to fit: shape = (nobs, batch_size) and
 *                          expects column major data layout. (device)
 * @param[in]  batch_size   Number of time series
 * @param[in]  nobs         Number of observations in a time series
 * @param[in]  p            Number of AR parameters
 * @param[in]  d            Difference order
 * @param[in]  q            Number of MA parameters
 * @param[in]  P            Number of seasonal AR parameters
 * @param[in]  D            Seasonal difference order
 * @param[in]  Q            Number of seasonal MA parameters
 * @param[in]  s            Seasonal period
 * @param[in]  intercept    Whether the model fits an intercept (constant term)
 * @param[in]  d_mu         mu if d != 0. Shape: (d, batch_size) (device)
 * @param[in]  d_ar         AR parameters. Shape: (p, batch_size) (device)
 * @param[in]  d_ma         MA parameters. Shape: (q, batch_size) (device)
 * @param[in]  d_sar        Seasonal AR parameters.
 *                          Shape: (P, batch_size) (device)
 * @param[in]  d_sma        Seasonal MA parameters.
 *                          Shape: (Q, batch_size) (device)
 * @param[in]  d_sigma2     Variance parameter. Shape: (batch_size,) (device)
 * @param[out] loglike      Log-Likelihood of the model per series (host)
 * @param[out] d_vs         The residual between model and original signal.
 *                          shape = (nobs, batch_size) (device)
 * @param[in]  trans        Run `jones_transform` on params.
 * @param[in]  host_loglike Whether loglike is a host pointer
 * @param[in]  fc_steps     Number of steps to forecast
 * @param[in]  d_fc         Array to store the forecast
 */
void batched_loglike(cumlHandle& handle, const double* d_y, int batch_size,
                     int nobs, int p, int d, int q, int P, int D, int Q, int s,
                     int intercept, const double* d_mu, const double* d_ar,
                     const double* d_ma, const double* d_sar,
                     const double* d_sma, const double* d_sigma2,
                     double* loglike, double* d_vs, bool trans = true,
                     bool host_loglike = true, int fc_steps = 0,
                     double* d_fc = nullptr);

/**
 * Batched in-sample and out-of-sample prediction of a time-series given all
 * the model parameters
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Batched Time series to predict.
 *                         Shape: (num_samples, batch size) (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  start       Index to start the prediction
 * @param[in]  end         Index to end the prediction (excluded)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference order
 * @param[in]  q           Number of MA parameters
 * @param[in]  P           Number of seasonal AR parameters
 * @param[in]  D           Seasonal difference order
 * @param[in]  Q           Number of seasonal MA parameters
 * @param[in]  s           Seasonal period
 * @param[in]  intercept   Whether the model fits an intercept (constant term)
 * @param[in]  d_params    Zipped trend, AR, and MA parameters
 *                         [mu, ar, ma] x batches (device)
 * @param[out] d_vs        Residual output (device)
 * @param[out] d_y_p       Prediction output (device)
 */
void predict(cumlHandle& handle, const double* d_y, int batch_size, int nobs,
             int start, int end, int p, int d, int q, int P, int D, int Q,
             int s, int intercept, const double* d_params, double* d_vs,
             double* d_y_p);

/**
 * Residual of in-sample prediction of a time-series given all the model
 * parameters.
 * 
 * @note: this overload should be used when the parameters are already unpacked
 *        to avoid useless packing / unpacking
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Batched Time series to predict.
 *                         Shape: (num_samples, batch size) (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference order
 * @param[in]  q           Number of MA parameters
 * @param[in]  P           Number of seasonal AR parameters
 * @param[in]  D           Seasonal difference order
 * @param[in]  Q           Number of seasonal MA parameters
 * @param[in]  s           Seasonal period
 * @param[in]  intercept   Whether the model fits an intercept (constant term)
 * @param[in]  d_mu        mu if intercept != 0. Shape: (d, batch_size) (device)
 * @param[in]  d_ar        AR parameters. Shape: (p, batch_size) (device)
 * @param[in]  d_ma        MA parameters. Shape: (q, batch_size) (device)
 * @param[in]  d_sar       Seasonal AR parameters.
 *                         Shape: (P, batch_size) (device)
 * @param[in]  d_sma       Seasonal MA parameters.
 *                         Shape: (Q, batch_size) (device)
 * @param[in]  d_sigma2    Variance parameter. Shape: (batch_size,) (device)
 * @param[out] d_vs        Residual output (device)
 * @param[in]  trans       Run `jones_transform` on params.
 * @param[in]  fc_steps    Number of steps to forecast
 * @param[in]  d_fc        Array to store the forecast
 */
void residual(cumlHandle& handle, const double* d_y, int batch_size, int nobs,
              int p, int d, int q, int P, int D, int Q, int s, int intercept,
              const double* d_mu, const double* d_ar, const double* d_sar,
              const double* d_sma, const double* d_sigma2, double* d_vs,
              bool trans, int fc_steps = 0, double* d_fc = nullptr);

/**
 * Compute an information criterion (AIC, AICc, BIC)
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Series to fit: shape = (nobs, batch_size) and
 *                         expects column major data layout. (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference order
 * @param[in]  q           Number of MA parameters
 * @param[in]  P           Number of seasonal AR parameters
 * @param[in]  D           Seasonal difference order
 * @param[in]  Q           Number of seasonal MA parameters
 * @param[in]  s           Seasonal period
 * @param[in]  intercept   Whether the model fits an intercept (constant term)
 * @param[in]  d_mu        mu if intercept != 0. Shape: (batch_size,) (device)
 * @param[in]  d_ar        AR parameters. Shape: (p, batch_size) (device)
 * @param[in]  d_ma        MA parameters. Shape: (q, batch_size) (device)
 * @param[in]  d_sar       Seasonal AR parameters.
 *                         Shape: (P, batch_size) (device)
 * @param[in]  d_sma       Seasonal MA parameters.
 *                         Shape: (Q, batch_size) (device)
 * @param[in]  d_sigma2    Variance parameter. Shape: (batch_size,) (device)
 * @param[out] ic          Array where to write the information criteria
 *                         Shape: (batch_size) (host)
 * @param[in]  ic_type     Type of information criterion wanted.
 *                         0: AIC, 1: AICc, 2: BIC
 */
void information_criterion(cumlHandle& handle, const double* d_y,
                           int batch_size, int nobs, int p, int d, int q, int P,
                           int D, int Q, int s, int intercept,
                           const double* d_mu, const double* d_ar,
                           const double* d_ma, const double* d_sar,
                           const double* d_sma, const double* d_sigma2,
                           double* ic, int ic_type);

/**
 * Provide initial estimates to ARIMA parameters mu, AR, and MA
 *
 * @param[in]  handle      cuML handle
 * @param[out] d_mu        mu if intercept != 0. Shape: (batch_size,) (device)
 * @param[out] d_ar        AR parameters. Shape: (p, batch_size) (device)
 * @param[out] d_ma        MA parameters. Shape: (q, batch_size) (device)
 * @param[out] d_sar       Seasonal AR parameters.
 *                         Shape: (P, batch_size) (device)
 * @param[out] d_sma       Seasonal MA parameters.
 *                         Shape: (Q, batch_size) (device)
 * @param[in]  d_sigma2    Variance parameter. Shape: (batch_size,) (device)
 * @param[in]  d_y         Series to fit: shape = (nobs, batch_size) and
 *                         expects column major data layout. (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference order
 * @param[in]  q           Number of MA parameters
 * @param[in]  P           Number of seasonal AR parameters
 * @param[in]  D           Seasonal difference order
 * @param[in]  Q           Number of seasonal MA parameters
 * @param[in]  s           Seasonal period
 * @param[in]  intercept   Whether the model fits an intercept (constant term)
 */
void estimate_x0(cumlHandle& handle, double* d_mu, double* d_ar, double* d_ma,
                 double* d_sar, double* d_sma, double* d_sigma2,
                 const double* d_y, int batch_size, int nobs, int p, int d,
                 int q, int P, int D, int Q, int s, int intercept);

}  // namespace ML
