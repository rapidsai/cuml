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

/* Compute the loglikelihood of the given parameter on the given time series in a batched context.
 * 
 * @param handle cuML handle
 * @param y series to fit: shape = (nobs, num_bathces) and expects column major data layout. Memory on Device.
 * @param num_batches number of time series
 * @param order ARIMA order (p: number of ar-parameters, d: difference parameter, q: number of ma-parameters)
 * @param params parameters to evaluate group by series: [mu0, ar.., ma.., mu1, ..] Memory on device.
 * @param d_vs The residual between model and original signal. shape = (nobs, num_batches) Memory on device.
 * @param trans run `jones_transform` on params.
 * @return vector of log likelihood, one for each series (size: num_batches). Memory on host.
 * @return kalman residual, shape = (nobs, num_batches) Memory on device.
 */
void batched_loglike(cumlHandle& handle, double* d_y, int num_batches, int nobs,
                     int p, int d, int q, double* d_params,
                     std::vector<double>& loglike, double* d_vs,
                     bool trans = true);

/* Batched in-sample prediction of a time-series given trend, AR, and MA parameters.
 * @param handle cuML handle
 * @param d_y Batched Time series to predict. Shape: (num_samples, batch size) (device)
 * @param num_batches Total number of batched time series
 * @param nobs Number of samples per time series (all series must be identical)
 * @param p Number of AR parameters
 * @param d Trend parameter
 * @param q Number of MA parameters
 * @param d_params Zipped trend, AR, and MA parameters [mu, ar, ma] x batches (device)
 * @param d_vs Residual output (device)
 * @param d_y_p Prediction output (device)
 */
void predict_in_sample(cumlHandle& handle, double* d_y, int num_batches,
                       int nobs, int p, int d, int q, double* d_params,
                       double* d_vs, double* d_y_p);

/* Residual of in-sample prediction of a time-series given trend, AR, and MA parameters.
 * @param handle cuML handle
 * @param d_y Batched Time series to predict. Shape: (num_samples, batch size) (device)
 * @param num_batches Total number of batched time series
 * @param nobs Number of samples per time series (all series must be identical)
 * @param p Number of AR parameters
 * @param d Trend parameter
 * @param q Number of MA parameters
 * @param d_params Zipped trend, AR, and MA parameters [mu, ar, ma] x batches (device)
 * @param d_vs Residual output (device)
 */
void residual(cumlHandle& handle, double* d_y, int num_batches, int nobs, int p,
              int d, int q, double* d_params, double* d_vs, bool trans);

/* Batched forecast of a time-series given trend, AR, and MA parameters.
 * @param handle cuML handle
 * @param num_steps The number of steps to forecast
 * @param p Number of AR parameters
 * @param d Trend parameter
 * @param q Number of MA parameters
 * @param batch_size Total number of batched time series
 * @param nobs Number of samples per time series (all series must be identical)
 * @param d_y Batched Time series to predict. Shape: (num_samples, batch size) (device)
 * @param d_y_diff Diffed (e.g., np.diff) of the batched Time series to
 * predict. Shape: (num_samples, batch size) (device)
 * @param d_vs Residual input (device)
 * @param d_params Zipped trend, AR, and MA parameters [mu, ar, ma] x batches (device)
 * @param d_y_fc Forecast output (device)
 */
void forecast(cumlHandle& handle, int num_steps, int p, int d, int q,
              int batch_size, int nobs, double* d_y, double* d_y_diff,
              double* d_vs, double* d_params, double* d_y_fc);

/**
 * TODO: doc
 * TODO: take mu, AR, MA as params. Host or device?
 * Provide initial estimates to ARIMA parameters mu, AR, and MA
 */
void estimate_x0(cumlHandle& handle, const double* d_y, int num_batches, int nobs, int p, int d,
                 int q);

}  // namespace ML
