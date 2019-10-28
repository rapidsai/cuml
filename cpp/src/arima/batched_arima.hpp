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
 * Compute the loglikelihood of the given parameter on the given time series in a batched context.
 *
 * @param[in]  handle      cuML handle
 * @param[in]  y           Series to fit: shape = (nobs, num_bathces) and expects column major data layout.
                           Memory on Device.
 * @param[in]  num_batches number of time series
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  params      Parameters to evaluate group by series: [mu0, ar.., ma.., mu1, ..]
                           Memory on device.
 * @param[out] loglike     Log-Likelihood of the model per series (host)
 * @param[out] d_vs        The residual between model and original signal. shape = (nobs, num_batches)
                           Memory on device.
 * @param[in]  trans       Run `jones_transform` on params.
 */
void batched_loglike(cumlHandle& handle, double* d_y, int num_batches, int nobs,
                     int p, int d, int q, double* d_params,
                     std::vector<double>& loglike, double* d_vs,
                     bool trans = true);

/**
 * Batched in-sample prediction of a time-series given trend, AR, and MA parameters.
 *
 * @param[in]  handle      cuML handle
 * @param[in]  d_y         Batched Time series to predict. Shape: (num_samples, batch size) (device)
 * @param[in]  num_batches Total number of batched time series
 * @param[in]  nobs        Number of samples per time series (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  d_params    Zipped trend, AR, and MA parameters [mu, ar, ma] x batches (device)
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
 * @param[in]  d_y         Batched Time series to predict. Shape: (num_samples, batch size) (device)
 * @param[in]  num_batches Total number of batched time series
 * @param[in]  nobs        Number of samples per time series (all series must be identical)
 * @param[in]  p           Number of AR parameters
 * @param[in]  d           Difference parameter
 * @param[in]  q           Number of MA parameters
 * @param[in]  d_params    Zipped trend, AR, and MA parameters [mu, ar, ma] x batches (device)
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
 * @param[in]  nobs       Number of samples per time series (all series must be identical)
 * @param[in]  d_y        Batched Time series to predict. Shape: (num_samples, batch size) (device)
 * @param[in]  d_y_diff   Diffed (e.g., np.diff) of the batched Time series to
 *                        predict. Shape: (num_samples, batch size) (device)
 * @param[in]  d_vs       Residual input (device)
 * @param[in]  d_params   Zipped trend, AR, and MA parameters [mu, ar, ma] x batches (device)
 * @param[out] d_y_fc     Forecast output (device)
 */
void forecast(cumlHandle& handle, int num_steps, int p, int d, int q,
              int batch_size, int nobs, double* d_y, double* d_y_diff,
              double* d_vs, double* d_params, double* d_y_fc);

}  // namespace ML
