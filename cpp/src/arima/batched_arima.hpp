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
 * @param[in]  d            Difference parameter
 * @param[in]  q            Number of MA parameters
 * @param[in]  intercept    Whether the model fits an intercept (constant term)
 * @param[in]  d_params     Parameters to evaluate group by series:
 *                          [mu0, ar.., ma.., mu1, ..] (device)
 * @param[out] loglike      Log-Likelihood of the model per series (host)
 * @param[out] d_vs         The residual between model and original signal.
 *                          shape = (nobs, batch_size) (device)
 * @param[in]  trans        Run `jones_transform` on params.
 * @param[in]  host_loglike Whether loglike is a host pointer
 */
void batched_loglike(cumlHandle& handle, double* d_y, int batch_size, int nobs,
                     int p, int d, int q, int intercept, double* d_params,
                     double* loglike, double* d_vs, bool trans = true,
                     bool host_loglike = true);

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
 * @param[in]  d            Difference parameter
 * @param[in]  q            Number of MA parameters
 * @param[in]  intercept    Whether the model fits an intercept (constant term)
 * @param[in]  d_mu         mu if d != 0. Shape: (d, batch_size) (device)
 * @param[in]  d_ar         AR parameters. Shape: (p, batch_size) (device)
 * @param[in]  d_ma         MA parameters. Shape: (q, batch_size) (device)
 * @param[out] loglike      Log-Likelihood of the model per series (host)
 * @param[out] d_vs         The residual between model and original signal.
 *                          shape = (nobs, batch_size) (device)
 * @param[in]  trans        Run `jones_transform` on params.
 * @param[in]  host_loglike Whether loglike is a host pointer
 */
void batched_loglike(cumlHandle& handle, double* d_y, int batch_size, int nobs,
                     int p, int d, int q, int intercept, double* d_mu,
                     double* d_ar, double* d_ma, double* loglike, double* d_vs,
                     bool trans = true, bool host_loglike = true);

/**
 * Batched in-sample prediction of a time-series given trend, AR, and MA
 * parameters.
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Batched Time series to predict.
 *                         Shape: (num_samples, batch size) (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  intercept   Whether the model fits an intercept (constant term)
 * @param[in]  d_params    Zipped trend, AR, and MA parameters
 *                         [mu, ar, ma] x batches (device)
 * @param[out] d_vs        Residual output (device)
 * @param[out] d_y_p       Prediction output (device)
 */
void predict_in_sample(cumlHandle& handle, double* d_y, int batch_size,
                       int nobs, int p, int d, int q, int intercept,
                       double* d_params, double* d_vs, double* d_y_p);

/**
 * Residual of in-sample prediction of a time-series given trend, AR, and MA parameters.
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Batched Time series to predict.
 *                         Shape: (num_samples, batch size) (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  intercept   Whether the model fits an intercept (constant term)
 * @param[in]  d_params    Zipped trend, AR, and MA parameters
 *                         [mu, ar, ma] x batches (device)
 * @param[out] d_vs        Residual output (device)
 * @param[in]  trans       Run `jones_transform` on params.
 */
void residual(cumlHandle& handle, double* d_y, int batch_size, int nobs, int p,
              int d, int q, int intercept, double* d_params, double* d_vs,
              bool trans);

///TODO: allow nullptr for the diff to be calculated inside this function if
///      needed (d > 0 or D > 0)
/**
 * Batched forecast of a time-series given trend, AR, and MA parameters.
 *
 * @param[in]  handle     cuML handle
 * @param[in]  num_steps  The number of steps to forecast
 * @param[in]  p          Number of AR parameters
 * @param[in]  d          Trend parameter
 * @param[in]  q          Number of MA parameters
 * @param[in]  intercept  Whether the model fits an intercept (constant term)
 * @param[in]  batch_size Total number of batched time series
 * @param[in]  nobs       Number of samples per time series
 *                        (all series must be identical)
 * @param[in]  d_y        Batched Time series to predict.
 *                        Shape: (num_samples, batch size) (device)
 * @param[in]  d_y_prep   Prepared data, or nullptr to prepare it now
 *                        Shape: (num_samples - d - D*s, batch size) (device)
 * @param[in]  d_vs       Residual input (device)
 * @param[in]  d_params   Zipped trend, AR, and MA parameters
 *                        [mu, ar, ma] x batches (device)
 * @param[out] d_y_fc     Forecast output (device)
 */
void forecast(cumlHandle& handle, int num_steps, int p, int d, int q,
              int intercept, int batch_size, int nobs, double* d_y,
              double* d_y_diff, double* d_vs, double* d_params, double* d_y_fc);

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
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  intercept   Whether the model fits an intercept (constant term)
 * @param[in]  d_mu        mu if d != 0. Shape: (d, batch_size) (device)
 * @param[in]  d_ar        AR parameters. Shape: (p, batch_size) (device)
 * @param[in]  d_ma        MA parameters. Shape: (q, batch_size) (device)
 * @param[out] ic          Array where to write the information criteria
 *                         Shape: (batch_size) (host)
 * @param[in]  ic_type     Type of information criterion wanted.
 *                         0: AIC, 1: AICc, 2: BIC
 */
void information_criterion(cumlHandle& handle, double* d_y, int batch_size,
                           int nobs, int p, int d, int q, int intercept,
                           double* d_mu, double* d_ar, double* d_ma, double* ic,
                           int ic_type);

/**
 * Provide initial estimates to ARIMA parameters mu, AR, and MA
 *
 * @param[in]  handle      cuML handle
 * @param[out] d_mu        mu if d != 0. Shape: (d, batch_size) (device)
 * @param[out] d_ar        AR parameters. Shape: (p, batch_size) (device)
 * @param[out] d_ma        MA parameters. Shape: (q, batch_size) (device)
 * @param[in]  d_y         Series to fit: shape = (nobs, batch_size) and
 *                         expects column major data layout. (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  intercept   Whether the model fits an intercept (constant term)
 */
void estimate_x0(cumlHandle& handle, double* d_mu, double* d_ar, double* d_ma,
                 const double* d_y, int batch_size, int nobs, int p, int d,
                 int q, int intercept);

}  // namespace ML
