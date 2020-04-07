/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <algorithm>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace ML {

/**
 * Structure to hold the ARIMA order (makes it easier to pass as an argument)
 */
struct ARIMAOrder {
  int p;  // Basic order
  int d;
  int q;
  int P;  // Seasonal order
  int D;
  int Q;
  int s;  // Seasonal period
  int k;  // Fit intercept?

  inline int r() const { return std::max(p + s * P, q + s * Q + 1); }
  inline int complexity() const { return p + P + q + Q + k + 1; }
  inline int lost_in_diff() const { return d + s * D; }

  inline bool need_prep() const { return static_cast<bool>(d + D + k); }
};

/**
 * Structure to hold the parameters (makes it easier to pass as an argument)
 * @note: the qualifier const applied to this structure will only guarantee
 *        that the pointers are not changed, but the user can still modify the
 *        arrays when using the pointers directly!
 */
template <typename DataT>
struct ARIMAParams {
  DataT* mu = nullptr;
  DataT* ar = nullptr;
  DataT* ma = nullptr;
  DataT* sar = nullptr;
  DataT* sma = nullptr;
  DataT* sigma2 = nullptr;

  /**
   * Allocate all the parameter device arrays
   *
   * @tparam      AllocatorT Type of allocator used
   * @param[in]   order      ARIMA order
   * @param[in]   batch_size Batch size
   * @param[in]   alloc      Allocator
   * @param[in]   stream     CUDA stream
   * @param[in]   order      ARIMA hyper-parameters
   * @param[in]   batch_size Number of time series analyzed
   * @param[in]   tr         Whether these are the transformed parameters
   */
  template <typename AllocatorT>
  void allocate(const ARIMAOrder& order, int batch_size, AllocatorT& alloc,
                cudaStream_t stream, bool tr = false) {
    if (order.k && !tr)
      mu = (DataT*)alloc->allocate(batch_size * sizeof(DataT), stream);
    if (order.p)
      ar =
        (DataT*)alloc->allocate(order.p * batch_size * sizeof(DataT), stream);
    if (order.q)
      ma =
        (DataT*)alloc->allocate(order.q * batch_size * sizeof(DataT), stream);
    if (order.P)
      sar =
        (DataT*)alloc->allocate(order.P * batch_size * sizeof(DataT), stream);
    if (order.Q)
      sma =
        (DataT*)alloc->allocate(order.Q * batch_size * sizeof(DataT), stream);
    sigma2 = (DataT*)alloc->allocate(batch_size * sizeof(DataT), stream);
  }

  /**
   * Deallocate all the parameter device arrays
   *
   * @tparam      AllocatorT Type of allocator used
   * @param[in]   order      ARIMA order
   * @param[in]   batch_size Batch size
   * @param[in]   alloc      Allocator
   * @param[in]   stream     CUDA stream
   * @param[in]   tr         Whether these are the transformed parameters
   */
  template <typename AllocatorT>
  void deallocate(const ARIMAOrder& order, int batch_size, AllocatorT& alloc,
                  cudaStream_t stream, bool tr = false) {
    if (order.k && !tr)
      alloc->deallocate(mu, batch_size * sizeof(DataT), stream);
    if (order.p)
      alloc->deallocate(ar, order.p * batch_size * sizeof(DataT), stream);
    if (order.q)
      alloc->deallocate(ma, order.q * batch_size * sizeof(DataT), stream);
    if (order.P)
      alloc->deallocate(sar, order.P * batch_size * sizeof(DataT), stream);
    if (order.Q)
      alloc->deallocate(sma, order.Q * batch_size * sizeof(DataT), stream);
    alloc->deallocate(sigma2, batch_size * sizeof(DataT), stream);
  }

  /**
   * Pack the separate parameter arrays into a unique parameter vector
   *
   * @param[in]   order      ARIMA order
   * @param[in]   batch_size Batch size
   * @param[out]  param_vec  Linear array of all parameters grouped by batch
   *                         [mu, ar, ma, sar, sma, sigma2] (device)
   * @param[in]  stream      CUDA stream
   */
  void pack(const ARIMAOrder& order, int batch_size, DataT* param_vec,
            cudaStream_t stream) const {
    int N = order.complexity();
    auto counting = thrust::make_counting_iterator(0);
    // The device lambda can't capture structure members...
    const DataT *_mu = mu, *_ar = ar, *_ma = ma, *_sar = sar, *_sma = sma,
                *_sigma2 = sigma2;
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       DataT* param = param_vec + bid * N;
                       if (order.k) {
                         *param = _mu[bid];
                         param++;
                       }
                       for (int ip = 0; ip < order.p; ip++) {
                         param[ip] = _ar[order.p * bid + ip];
                       }
                       param += order.p;
                       for (int iq = 0; iq < order.q; iq++) {
                         param[iq] = _ma[order.q * bid + iq];
                       }
                       param += order.q;
                       for (int iP = 0; iP < order.P; iP++) {
                         param[iP] = _sar[order.P * bid + iP];
                       }
                       param += order.P;
                       for (int iQ = 0; iQ < order.Q; iQ++) {
                         param[iQ] = _sma[order.Q * bid + iQ];
                       }
                       param += order.Q;
                       *param = _sigma2[bid];
                     });
  }

  /**
   * Unpack a parameter vector into separate arrays of parameters.
   * 
   * @param[in]  order      ARIMA order
   * @param[in]  batch_size Batch size
   * @param[in]  param_vec  Linear array of all parameters grouped by batch
   *                        [mu, ar, ma, sar, sma, sigma2] (device)
   * @param[in]  stream     CUDA stream
   */
  void unpack(const ARIMAOrder& order, int batch_size, const DataT* param_vec,
              cudaStream_t stream) {
    int N = order.complexity();
    auto counting = thrust::make_counting_iterator(0);
    // The device lambda can't capture structure members...
    DataT *_mu = mu, *_ar = ar, *_ma = ma, *_sar = sar, *_sma = sma,
          *_sigma2 = sigma2;
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       const DataT* param = param_vec + bid * N;
                       if (order.k) {
                         _mu[bid] = *param;
                         param++;
                       }
                       for (int ip = 0; ip < order.p; ip++) {
                         _ar[order.p * bid + ip] = param[ip];
                       }
                       param += order.p;
                       for (int iq = 0; iq < order.q; iq++) {
                         _ma[order.q * bid + iq] = param[iq];
                       }
                       param += order.q;
                       for (int iP = 0; iP < order.P; iP++) {
                         _sar[order.P * bid + iP] = param[iP];
                       }
                       param += order.P;
                       for (int iQ = 0; iQ < order.Q; iQ++) {
                         _sma[order.Q * bid + iQ] = param[iQ];
                       }
                       param += order.Q;
                       _sigma2[bid] = *param;
                     });
  }
};

}  // namespace ML
