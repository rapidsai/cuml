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
 * @param[in]  p               Number of AR parameters
 * @param[in]  q               Number of MA parameters
 * @param[in]  P               Number of seasonal AR parameters
 * @param[in]  Q               Number of seasonal MA parameters
 * @param[in]  batch_size      Number of series making up the batch
 * @param[out] loglike_b       Resulting loglikelihood (for each series)
 * @param[out] d_vs            Residual between the prediction and the
 *                             original series.
 *                             shape=(nobs, batch_size) (device)
 * @param[in]  host_loglike    Whether loglike is a host pointer
 * @param[in]  initP_kalman_it Initialize the Kalman filter covariance `P`
 *                             with 1 or more kalman iterations instead of
 *                             an analytical heuristic.
 */
void batched_kalman_filter(cumlHandle& handle, const double* d_ys_b, int nobs,
                           const double* d_ar, const double* d_ma,
                           const double* d_sar, const double* d_sma, int p,
                           int q, int P, int Q, int batch_size, double* loglike,
                           double* d_vs, bool host_loglike = true,
                           bool initP_kalman_it = false);

/**
 * Turns linear array of parameters into arrays of mu, ar, and ma parameters.
 * (using device arrays)
 * 
 * @param[in]  d_params   Linear array of all parameters grouped by batch
 *                        [mu, ar, ma] (device)
 * @param[out] d_mu       Trend parameter (device)
 * @param[out] d_ar       AR parameters (device)
 * @param[out] d_ma       MA parameters (device)
 * @param[out] d_sar      Seasonal AR parameters (device)
 * @param[out] d_sma      Seasonal MA parameters (device)
 * @param[in]  batch_size Number of time series analyzed.
 * @param[in]  p          Number of AR parameters
 * @param[in]  q          Number of MA parameters
 * @param[in]  P          Number of seasonal AR parameters
 * @param[in]  Q          Number of seasonal MA parameters
 * @param[in]  k          Whether the model fits an intercept
 * @param[in]  stream     CUDA stream
 */
void unpack(const double* d_params, double* d_mu, double* d_ar, double* d_ma,
            double* d_sar, double* d_sma, int batch_size, int p, int q, int P,
            int Q, int k, cudaStream_t stream);

/**
 * Helper function to allocate all the parameter device arrays
 *
 * @tparam      AllocatorT Type of allocator used
 * @param[in]   al         Allocator
 * @param[in]   stream     CUDA stream
 * @param[in]   p          Number of AR parameters
 * @param[in]   q          Number of MA parameters
 * @param[in]   P          Number of seasonal AR parameters
 * @param[in]   Q          Number of seasonal MA parameters
 * @param[in]   batch_size Number of time series analyzed.
 * @param[out]  d_ar       AR parameters to allocate (device)
 * @param[out]  d_ma       MA parameters to allocate (device)
 * @param[out]  d_sar      Seasonal AR parameters to allocate (device)
 * @param[out]  d_sma      Seasonal MA parameters to allocate (device)
 * @param[in]   k          Whether to fit an intercept
 * @param[out]  d_mu       Intercept parameters to allocate (device)
 */
template <typename AllocatorT>
void allocate_params(AllocatorT& alloc, cudaStream_t stream, int p, int q,
                     int P, int Q, int batch_size, double** d_ar, double** d_ma,
                     double** d_sar, double** d_sma, int k = 0,
                     double** d_mu = nullptr);

/**
 * Helper function to deallocate all the parameter device arrays
 *
 * @tparam      AllocatorT Type of allocator used
 * @param[in]   al         Allocator
 * @param[in]   stream     CUDA stream
 * @param[in]   p          Number of AR parameters
 * @param[in]   q          Number of MA parameters
 * @param[in]   P          Number of seasonal AR parameters
 * @param[in]   Q          Number of seasonal MA parameters
 * @param[in]   batch_size Number of time series analyzed.
 * @param[out]  d_ar       AR parameters to deallocate (device)
 * @param[out]  d_ma       MA parameters to deallocate (device)
 * @param[out]  d_sar      Seasonal AR parameters to deallocate (device)
 * @param[out]  d_sma      Seasonal MA parameters to deallocate (device)
 * @param[in]   k          Whether to fit an intercept
 * @param[out]  d_mu       Intercept parameters to deallocate (device)
 */
template <typename AllocatorT>
void deallocate_params(AllocatorT& alloc, cudaStream_t stream, int p, int q,
                       int P, int Q, int batch_size, double* d_ar, double* d_ma,
                       double* d_sar, double* d_sma, int k = 0,
                       double* d_mu = nullptr);

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
