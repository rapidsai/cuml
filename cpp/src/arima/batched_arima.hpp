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
 * @param[in]  d_y          Series to fit: shape = (nobs, num_batches) and
 *                          expects column major data layout. (device)
 * @param[in]  num_batches  Number of time series
 * @param[in]  nobs         Number of observations in a time series
 * @param[in]  p            Number of AR parameters
 * @param[in]  d            Difference parameter
 * @param[in]  q            Number of MA parameters
 * @param[in]  d_params     Parameters to evaluate group by series:
 *                          [mu0, ar.., ma.., mu1, ..] (device)
 * @param[out] loglike      Log-Likelihood of the model per series (host)
 * @param[out] d_vs         The residual between model and original signal.
 *                          shape = (nobs, num_batches) (device)
 * @param[in]  trans        Run `jones_transform` on params.
 * @param[in]  host_loglike Whether loglike is a host pointer
 */
void batched_loglike(cumlHandle& handle, double* d_y, int num_batches, int nobs,
                     int p, int d, int q, double* d_params,
                     double* loglike, double* d_vs,
                     bool trans = true, bool host_loglike = true);

/**
 * Compute the loglikelihood of the given parameter on the given time series
 * in a batched context.
 * 
 * @note: this overload should be used when the parameters are already unpacked
 *        to avoid useless packing / unpacking
 *
 * @param[in]  handle       cuML handle
 * @param[in]  d_y          Series to fit: shape = (nobs, num_batches) and
 *                          expects column major data layout. (device)
 * @param[in]  num_batches  Number of time series
 * @param[in]  nobs         Number of observations in a time series
 * @param[in]  p            Number of AR parameters
 * @param[in]  d            Difference parameter
 * @param[in]  q            Number of MA parameters
 * @param[in]  d_mu         mu if d != 0. Shape: (d, num_batches) (device)
 * @param[in]  d_ar         AR parameters. Shape: (p, num_batches) (device)
 * @param[in]  d_ma         MA parameters. Shape: (q, num_batches) (device)
 * @param[out] loglike      Log-Likelihood of the model per series (host)
 * @param[out] d_vs         The residual between model and original signal.
 *                          shape = (nobs, num_batches) (device)
 * @param[in]  trans        Run `jones_transform` on params.
 * @param[in]  host_loglike Whether loglike is a host pointer
 */
void batched_loglike(cumlHandle& handle, double* d_y, int num_batches, int nobs,
                     int p, int d, int q, double* d_mu, double* d_ar,
                     double* d_ma, double* loglike, double* d_vs,
                     bool trans = true, bool host_loglike = true);

/**
 * Batched in-sample prediction of a time-series given trend, AR, and MA
 * parameters.
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Batched Time series to predict.
 *                         Shape: (num_samples, batch size) (device)
 * @param[in]  num_batches Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  d_params    Zipped trend, AR, and MA parameters
 *                         [mu, ar, ma] x batches (device)
 * @param[out] d_vs        Residual output (device)
 * @param[out] d_y_p       Prediction output (device)
 */
void predict_in_sample(cumlHandle& handle, double* d_y, int num_batches,
                       int nobs, int p, int d, int q, double* d_params,
                       double* d_vs, double* d_y_p);

/**
 * Residual of in-sample prediction of a time-series given trend, AR, and MA parameters.
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Batched Time series to predict.
 *                         Shape: (num_samples, batch size) (device)
 * @param[in]  num_batches Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  d_params    Zipped trend, AR, and MA parameters
 *                         [mu, ar, ma] x batches (device)
 * @param[out] d_vs        Residual output (device)
 * @param[in]  trans       Run `jones_transform` on params.
 */
void residual(cumlHandle& handle, double* d_y, int num_batches, int nobs, int p,
              int d, int q, double* d_params, double* d_vs, bool trans);

/**
 * Batched forecast of a time-series given trend, AR, and MA parameters.
 *
 * @param[in]  handle     cuML handle
 * @param[in]  num_steps  The number of steps to forecast
 * @param[in]  p          Number of AR parameters
 * @param[in]  d          Trend parameter
 * @param[in]  q          Number of MA parameters
 * @param[in]  batch_size Total number of batched time series
 * @param[in]  nobs       Number of samples per time series
 *                        (all series must be identical)
 * @param[in]  d_y        Batched Time series to predict.
 *                        Shape: (num_samples, batch size) (device)
 * @param[in]  d_y_diff   Diffed (e.g., np.diff) of the batched Time series to
 *                        predict. Shape: (num_samples, batch size) (device)
 * @param[in]  d_vs       Residual input (device)
 * @param[in]  d_params   Zipped trend, AR, and MA parameters
 *                        [mu, ar, ma] x batches (device)
 * @param[out] d_y_fc     Forecast output (device)
 */
void forecast(cumlHandle& handle, int num_steps, int p, int d, int q,
              int batch_size, int nobs, double* d_y, double* d_y_diff,
              double* d_vs, double* d_params, double* d_y_fc);

/**
 * Compute Akaike information criterion.
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Series to fit: shape = (nobs, num_batches) and
 *                         expects column major data layout. (device)
 * @param[in]  num_batches Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  d_mu        mu if d != 0. Shape: (d, num_batches) (device)
 * @param[in]  d_ar        AR parameters. Shape: (p, num_batches) (device)
 * @param[in]  d_ma        MA parameters. Shape: (q, num_batches) (device)
 * @param[out] ic          Array where to write the information criteria
 *                         Shape: (num_batches) (host)
 */
void aic(cumlHandle& handle, double* d_y, int num_batches, int nobs, int p,
         int d, int q, double* d_mu, double* d_ar, double* d_ma, double* ic);

/**
 * Compute corrected Akaike information criterion.
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Series to fit: shape = (nobs, num_batches) and
 *                         expects column major data layout. (device)
 * @param[in]  num_batches Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  d_mu        mu if d != 0. Shape: (d, num_batches) (device)
 * @param[in]  d_ar        AR parameters. Shape: (p, num_batches) (device)
 * @param[in]  d_ma        MA parameters. Shape: (q, num_batches) (device)
 * @param[out] ic          Array where to write the information criteria
 *                         Shape: (num_batches) (host)
 */
void aicc(cumlHandle& handle, double* d_y, int num_batches, int nobs, int p,
          int d, int q, double* d_mu, double* d_ar, double* d_ma, double* ic);

/**
 * Compute Bayesian information criterion.
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Series to fit: shape = (nobs, num_batches) and
 *                         expects column major data layout. (device)
 * @param[in]  num_batches Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  d_mu        mu if d != 0. Shape: (d, num_batches) (device)
 * @param[in]  d_ar        AR parameters. Shape: (p, num_batches) (device)
 * @param[in]  d_ma        MA parameters. Shape: (q, num_batches) (device)
 * @param[out] ic          Array where to write the information criteria
 *                         Shape: (num_batches) (host)
 */
void bic(cumlHandle& handle, double* d_y, int num_batches, int nobs, int p,
         int d, int q, double* d_mu, double* d_ar, double* d_ma, double* ic);

/**
 * Provide initial estimates to ARIMA parameters mu, AR, and MA
 *
 * @param[in]  handle      cuML handle
 * @param[out] d_mu        mu if d != 0. Shape: (d, num_batches) (device)
 * @param[out] d_ar        AR parameters. Shape: (p, num_batches) (device)
 * @param[out] d_ma        MA parameters. Shape: (q, num_batches) (device)
 * @param[in]  d_y         Series to fit: shape = (nobs, num_batches) and
 *                         expects column major data layout. (device)
 * @param[in]  num_batches Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 */
void estimate_x0(cumlHandle& handle, double* d_mu, double* d_ar, double* d_ma,
                 const double* d_y, int num_batches, int nobs, int p, int d,
                 int q);

}  // namespace ML
