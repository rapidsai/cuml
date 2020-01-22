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

#include <cuda_runtime.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

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
 * @param[in]   d_sigma2   Variance parameters to allocate (device)
 *                         Ignored if tr == true
 * @param[in]   tr         Whether these are the transformed parameters
 * @param[in]   k          Whether to fit an intercept
 * @param[out]  d_mu       Intercept parameters to allocate (device)
 */
template <typename AllocatorT>
static void allocate_params(AllocatorT& alloc, cudaStream_t stream, int p,
                            int q, int P, int Q, int batch_size, double** d_ar,
                            double** d_ma, double** d_sar, double** d_sma,
                            double** d_sigma2, bool tr = false, int k = 0,
                            double** d_mu = nullptr) {
  if (k) *d_mu = (double*)alloc->allocate(batch_size * sizeof(double), stream);
  if (p)
    *d_ar = (double*)alloc->allocate(p * batch_size * sizeof(double), stream);
  if (q)
    *d_ma = (double*)alloc->allocate(q * batch_size * sizeof(double), stream);
  if (P)
    *d_sar = (double*)alloc->allocate(P * batch_size * sizeof(double), stream);
  if (Q)
    *d_sma = (double*)alloc->allocate(Q * batch_size * sizeof(double), stream);
  if (!tr)
    *d_sigma2 = (double*)alloc->allocate(batch_size * sizeof(double), stream);
}

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
 * @param[out]  d_sigma2   Variance parameters to deallocate (device)
 *                         Ignored if tr == true
 * @param[in]   tr         Whether these are the transformed parameters
 * @param[in]   k          Whether to fit an intercept
 * @param[out]  d_mu       Intercept parameters to deallocate (device)
 */
template <typename AllocatorT>
static void deallocate_params(AllocatorT& alloc, cudaStream_t stream, int p,
                              int q, int P, int Q, int batch_size, double* d_ar,
                              double* d_ma, double* d_sar, double* d_sma,
                              double* d_sigma2, bool tr = false, int k = 0,
                              double* d_mu = nullptr) {
  if (k) alloc->deallocate(d_mu, batch_size * sizeof(double), stream);
  if (p) alloc->deallocate(d_ar, p * batch_size * sizeof(double), stream);
  if (q) alloc->deallocate(d_ma, q * batch_size * sizeof(double), stream);
  if (P) alloc->deallocate(d_sar, P * batch_size * sizeof(double), stream);
  if (Q) alloc->deallocate(d_sma, Q * batch_size * sizeof(double), stream);
  if (!tr) alloc->deallocate(d_sigma2, batch_size * sizeof(double), stream);
}

/**
 * Helper function to pack the separate parameter arrays into a unique
 * parameter vector
 *
 * @param[in]  batch_size  Batch size
 * @param[in]  p           Number of AR parameters
 * @param[in]  q           Number of MA parameters
 * @param[in]  P           Number of seasonal AR parameters
 * @param[in]  Q           Number of seasonal MA parameters
 * @param[in]  k           Whether the model fits an intercept (constant term)
 * @param[in]  d_mu        mu if intercept != 0. Shape: (batch_size,) (device)
 * @param[in]  d_ar        AR parameters. Shape: (p, batch_size) (device)
 * @param[in]  d_ma        MA parameters. Shape: (q, batch_size) (device)
 * @param[in]  d_sar       Seasonal AR parameters.
 *                         Shape: (P, batch_size) (device)
 * @param[in]  d_sma       Seasonal MA parameters.
 *                         Shape: (Q, batch_size) (device)
 * @param[in]  d_sigma2    Variance parameters. Shape: (batch_size,) (device)
 * @param[out] d_params    Output parameter vector
 * @param[in]  stream      CUDA stream
 */
static void pack(int batch_size, int p, int q, int P, int Q, int k,
                 const double* d_mu, const double* d_ar, const double* d_ma,
                 const double* d_sar, const double* d_sma,
                 const double* d_sigma2, double* d_params,
                 cudaStream_t stream) {
  int N = p + q + P + Q + k + 1;
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     double* param = d_params + bid * N;
                     if (k) {
                       *param = d_mu[bid];
                       param++;
                     }
                     for (int ip = 0; ip < p; ip++) {
                       param[ip] = d_ar[p * bid + ip];
                     }
                     param += p;
                     for (int iq = 0; iq < q; iq++) {
                       param[iq] = d_ma[q * bid + iq];
                     }
                     param += q;
                     for (int iP = 0; iP < P; iP++) {
                       param[iP] = d_sar[P * bid + iP];
                     }
                     param += P;
                     for (int iQ = 0; iQ < Q; iQ++) {
                       param[iQ] = d_sma[Q * bid + iQ];
                     }
                     param += Q;
                     *param = d_sigma2[bid];
                   });
}

/**
 * Helper function to unpack a linear array of parameters into separate arrays
 * of parameters.
 * 
 * @param[in]  d_params   Linear array of all parameters grouped by batch
 *                        [mu, ar, ma] (device)
 * @param[out] d_mu       mu if intercept != 0. Shape: (batch_size,) (device)
 * @param[out] d_ar       AR parameters. Shape: (p, batch_size) (device)
 * @param[out] d_ma       MA parameters. Shape: (q, batch_size) (device)
 * @param[out] d_sar      Seasonal AR parameters.
 *                        Shape: (P, batch_size) (device)
 * @param[out] d_sma      Seasonal MA parameters.
 *                        Shape: (Q, batch_size) (device)
 * @param[out] d_sigma2   Variance parameters. Shape: (batch_size,) (device)
 * @param[in]  batch_size Number of time series analyzed.
 * @param[in]  p          Number of AR parameters
 * @param[in]  q          Number of MA parameters
 * @param[in]  P          Number of seasonal AR parameters
 * @param[in]  Q          Number of seasonal MA parameters
 * @param[in]  k          Whether the model fits an intercept
 * @param[in]  stream     CUDA stream
 */
static void unpack(const double* d_params, double* d_mu, double* d_ar,
                   double* d_ma, double* d_sar, double* d_sma, double* d_sigma2,
                   int batch_size, int p, int q, int P, int Q, int k,
                   cudaStream_t stream) {
  int N = p + q + P + Q + k + 1;
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     const double* param = d_params + bid * N;
                     if (k) {
                       d_mu[bid] = *param;
                       param++;
                     }
                     for (int ip = 0; ip < p; ip++) {
                       d_ar[p * bid + ip] = param[ip];
                     }
                     param += p;
                     for (int iq = 0; iq < q; iq++) {
                       d_ma[q * bid + iq] = param[iq];
                     }
                     param += q;
                     for (int iP = 0; iP < P; iP++) {
                       d_sar[P * bid + iP] = param[iP];
                     }
                     param += P;
                     for (int iQ = 0; iQ < Q; iQ++) {
                       d_sma[Q * bid + iQ] = param[iQ];
                     }
                     param += Q;
                     d_sigma2[bid] = *param;
                   });
}

}  // namespace ML
