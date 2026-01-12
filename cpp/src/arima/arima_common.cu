/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/tsa/arima_common.h>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace ML {

template <typename DataT>
void ARIMAParams<DataT>::pack(const ARIMAOrder& order,
                              int batch_size,
                              DataT* param_vec,
                              cudaStream_t stream) const
{
  int N         = order.complexity();
  auto counting = thrust::make_counting_iterator(0);
  // The device lambda can't capture structure members...
  const DataT *_mu = mu, *_beta = beta, *_ar = ar, *_ma = ma, *_sar = sar, *_sma = sma,
              *_sigma2 = sigma2;
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
      DataT* param = param_vec + bid * N;
      if (order.k) {
        *param = _mu[bid];
        param++;
      }
      for (int i = 0; i < order.n_exog; i++) {
        param[i] = _beta[order.n_exog * bid + i];
      }
      param += order.n_exog;
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

template <typename DataT>
void ARIMAParams<DataT>::unpack(const ARIMAOrder& order,
                                int batch_size,
                                const DataT* param_vec,
                                cudaStream_t stream)
{
  int N         = order.complexity();
  auto counting = thrust::make_counting_iterator(0);
  // The device lambda can't capture structure members...
  DataT *_mu = mu, *_beta = beta, *_ar = ar, *_ma = ma, *_sar = sar, *_sma = sma, *_sigma2 = sigma2;
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int bid) {
      const DataT* param = param_vec + bid * N;
      if (order.k) {
        _mu[bid] = *param;
        param++;
      }
      for (int i = 0; i < order.n_exog; i++) {
        _beta[order.n_exog * bid + i] = param[i];
      }
      param += order.n_exog;
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

// Explicit template instantiation
template struct ARIMAParams<double>;

}  // namespace ML
