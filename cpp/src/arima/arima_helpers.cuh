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

#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "arima_common.h"

namespace ML {

/**
 * Auxiliary function of reduced_polynomial. Computes a coefficient of an (S)AR
 * or (S)MA polynomial based on the values of the corresponding parameters
 *
 * @tparam     isAr    Is this an AR (true) or MA (false) polynomial?
 * @param[in]  param   Parameter array
 * @param[in]  lags    Number of parameters
 * @param[in]  idx     Which coefficient to compute
 * @return             The value of the coefficient
 */
template <bool isAr>
static inline __host__ __device__ double _param_to_poly(const double* param,
                                                        int lags, int idx) {
  if (idx > lags) {
    return 0.0;
  } else if (idx) {
    return isAr ? -param[idx - 1] : param[idx - 1];
  } else
    return 1.0;
}

/**
 * Helper function to compute the reduced AR or MA polynomial based on the
 * AR and SAR or MA and SMA parameters
 *
 * @tparam     isAr    Is this an AR (true) or MA (false) polynomial?
 * @param[in]  bid     Batch id
 * @param[in]  param   Non-seasonal parameters
 * @param[in]  lags    Number of non-seasonal parameters
 * @param[in]  sparam  Seasonal parameters
 * @param[in]  slags   Number of seasonal parameters
 * @param[in]  s       Seasonal period
 * @param[in]  idx     Which coefficient to compute
 * @return             The value of the coefficient
 */
template <bool isAr>
static inline __host__ __device__ double reduced_polynomial(
  int bid, const double* param, int lags, const double* sparam, int slags,
  int s, int idx) {
  int idx1 = s ? idx / s : 0;
  int idx0 = idx - s * idx1;
  double coef0 = _param_to_poly<isAr>(param + bid * lags, lags, idx0);
  double coef1 = _param_to_poly<isAr>(sparam + bid * slags, slags, idx1);
  return isAr ? -coef0 * coef1 : coef0 * coef1;
}

/**
 * Helper function to allocate all the parameter device arrays
 *
 * @tparam      AllocatorT Type of allocator used
 * @param[in]   alloc      Allocator
 * @param[in]   stream     CUDA stream
 * @param[in]   order      ARIMA hyper-parameters
 * @param[in]   batch_size Number of time series analyzed.
 * @param[in]   params     ARIMA parameters (device)
 * @param[in]   tr         Whether these are the transformed parameters
 */
template <typename AllocatorT>
static void allocate_params(AllocatorT& alloc, cudaStream_t stream,
                            ARIMAOrder order, int batch_size,
                            ARIMAParamsD* params, bool tr = false) {
  if (order.k && !tr)
    params->mu = (double*)alloc->allocate(batch_size * sizeof(double), stream);
  if (order.p)
    params->ar =
      (double*)alloc->allocate(order.p * batch_size * sizeof(double), stream);
  if (order.q)
    params->ma =
      (double*)alloc->allocate(order.q * batch_size * sizeof(double), stream);
  if (order.P)
    params->sar =
      (double*)alloc->allocate(order.P * batch_size * sizeof(double), stream);
  if (order.Q)
    params->sma =
      (double*)alloc->allocate(order.Q * batch_size * sizeof(double), stream);
  if (!tr)
    params->sigma2 =
      (double*)alloc->allocate(batch_size * sizeof(double), stream);
}

/**
 * Helper function to deallocate all the parameter device arrays
 *
 * @tparam      AllocatorT Type of allocator used
 * @param[in]   alloc      Allocator
 * @param[in]   stream     CUDA stream
 * @param[in]   order      ARIMA hyper-parameters
 * @param[in]   batch_size Number of time series analyzed
 * @param[in]   params     ARIMA parameters (device)
 * @param[in]   tr         Whether these are the transformed parameters
 */
template <typename AllocatorT>
static void deallocate_params(AllocatorT& alloc, cudaStream_t stream,
                              ARIMAOrder order, int batch_size,
                              ARIMAParamsD params, bool tr = false) {
  if (order.k && !tr)
    alloc->deallocate(params.mu, batch_size * sizeof(double), stream);
  if (order.p)
    alloc->deallocate(params.ar, order.p * batch_size * sizeof(double), stream);
  if (order.q)
    alloc->deallocate(params.ma, order.q * batch_size * sizeof(double), stream);
  if (order.P)
    alloc->deallocate(params.sar, order.P * batch_size * sizeof(double),
                      stream);
  if (order.Q)
    alloc->deallocate(params.sma, order.Q * batch_size * sizeof(double),
                      stream);
  if (!tr)
    alloc->deallocate(params.sigma2, batch_size * sizeof(double), stream);
}

/**
 * Helper function to pack the separate parameter arrays into a unique
 * parameter vector
 *
 * @param[in]  batch_size  Batch size
 * @param[in]  order       ARIMA hyper-parameters
 * @param[in]  params      ARIMA parameters (device)
 * @param[out] param_vec   Output parameter vector
 * @param[in]  stream      CUDA stream
 */
static void pack(int batch_size, ARIMAOrder order, const ARIMAParamsD params,
                 double* param_vec, cudaStream_t stream) {
  int N = order.complexity();
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     double* param = param_vec + bid * N;
                     if (order.k) {
                       *param = params.mu[bid];
                       param++;
                     }
                     for (int ip = 0; ip < order.p; ip++) {
                       param[ip] = params.ar[order.p * bid + ip];
                     }
                     param += order.p;
                     for (int iq = 0; iq < order.q; iq++) {
                       param[iq] = params.ma[order.q * bid + iq];
                     }
                     param += order.q;
                     for (int iP = 0; iP < order.P; iP++) {
                       param[iP] = params.sar[order.P * bid + iP];
                     }
                     param += order.P;
                     for (int iQ = 0; iQ < order.Q; iQ++) {
                       param[iQ] = params.sma[order.Q * bid + iQ];
                     }
                     param += order.Q;
                     *param = params.sigma2[bid];
                   });
}

/**
 * Helper function to unpack a linear array of parameters into separate arrays
 * of parameters.
 * 
 * @param[in]  param_vec  Linear array of all parameters grouped by batch
 *                        [mu, ar, ma, sar, sma, sigma2] (device)
 * @param[out] params     ARIMA parameters (device)
 * @param[in]  batch_size Number of time series analyzed.
 * @param[in]  order      ARIMA hyper-parameters
 * @param[in]  stream     CUDA stream
 */
static void unpack(const double* param_vec, ARIMAParamsD params, int batch_size,
                   ARIMAOrder order, cudaStream_t stream) {
  int N = order.complexity();
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     const double* param = param_vec + bid * N;
                     if (order.k) {
                       params.mu[bid] = *param;
                       param++;
                     }
                     for (int ip = 0; ip < order.p; ip++) {
                       params.ar[order.p * bid + ip] = param[ip];
                     }
                     param += order.p;
                     for (int iq = 0; iq < order.q; iq++) {
                       params.ma[order.q * bid + iq] = param[iq];
                     }
                     param += order.q;
                     for (int iP = 0; iP < order.P; iP++) {
                       params.sar[order.P * bid + iP] = param[iP];
                     }
                     param += order.P;
                     for (int iQ = 0; iQ < order.Q; iQ++) {
                       params.sma[order.Q * bid + iQ] = param[iQ];
                     }
                     param += order.Q;
                     params.sigma2[bid] = *param;
                   });
}

}  // namespace ML
