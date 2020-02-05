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

#include "cuda_utils.h"
#include "cuml/tsa/arima_common.h"
#include "linalg/matrix_vector_op.h"
#include "timeSeries/jones_transform.h"

namespace MLCommon {
namespace TimeSeries {

/**
 * Auxiliary function of reduced_polynomial. Computes a coefficient of an (S)AR
 * or (S)MA polynomial based on the values of the corresponding parameters
 *
 * @tparam     isAr    Is this an AR (true) or MA (false) polynomial?
 * @tparam     DataT   Scalar type
 * @param[in]  param   Parameter array
 * @param[in]  lags    Number of parameters
 * @param[in]  idx     Which coefficient to compute
 * @return             The value of the coefficient
 */
template <bool isAr, typename DataT>
HDI DataT _param_to_poly(const DataT* param, int lags, int idx) {
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
 * @tparam     DataT   Scalar type
 * @param[in]  bid     Batch id
 * @param[in]  param   Non-seasonal parameters
 * @param[in]  lags    Number of non-seasonal parameters
 * @param[in]  sparam  Seasonal parameters
 * @param[in]  slags   Number of seasonal parameters
 * @param[in]  s       Seasonal period
 * @param[in]  idx     Which coefficient to compute
 * @return             The value of the coefficient
 */
template <bool isAr, typename DataT>
HDI DataT reduced_polynomial(int bid, const DataT* param, int lags,
                             const DataT* sparam, int slags, int s, int idx) {
  int idx1 = s ? idx / s : 0;
  int idx0 = idx - s * idx1;
  DataT coef0 = _param_to_poly<isAr>(param + bid * lags, lags, idx0);
  DataT coef1 = _param_to_poly<isAr>(sparam + bid * slags, slags, idx1);
  return isAr ? -coef0 * coef1 : coef0 * coef1;
}

/**
 * @brief Prepare data by differencing if needed (simple and/or seasonal)
 *        and removing a trend if needed
 *
 * @note: It is assumed that d + D <= 2. This is enforced on the Python side
 *
 * @param[out] d_out       Output. Shape (n_obs - d - D*s, batch_size) (device)
 * @param[in]  d_in        Input. Shape (n_obs, batch_size) (device)
 * @param[in]  batch_size  Number of series per batch
 * @param[in]  n_obs       Number of observations per series
 * @param[in]  d           Order of simple differences (0, 1 or 2)
 * @param[in]  D           Order of seasonal differences (0, 1 or 2)
 * @param[in]  s           Seasonal period if D > 0
 * @param[in]  stream      CUDA stream
 * @param[in]  intercept   Whether the model fits an intercept
 * @param[in]  d_mu        Mu array if intercept > 0
 *                         Shape (batch_size,) (device)
 */
template <typename DataT>
static void prepare_data(DataT* d_out, const DataT* d_in, int batch_size,
                         int n_obs, int d, int D, int s, cudaStream_t stream,
                         int intercept = 0, const DataT* d_mu = nullptr) {
  // Only one difference (simple or seasonal)
  if (d + D == 1) {
    int period = d ? 1 : s;
    int tpb = (n_obs - period) > 512 ? 256 : 128;  // quick heuristics
    MLCommon::LinAlg::Batched::
      batched_diff_kernel<<<batch_size, tpb, 0, stream>>>(d_in, d_out, n_obs,
                                                          period);
    CUDA_CHECK(cudaPeekAtLastError());
  }
  // Two differences (simple or seasonal or both)
  else if (d + D == 2) {
    int period1 = d ? 1 : s;
    int period2 = d == 2 ? 1 : s;
    int tpb = (n_obs - period1 - period2) > 512 ? 256 : 128;
    MLCommon::LinAlg::Batched::
      batched_second_diff_kernel<<<batch_size, tpb, 0, stream>>>(
        d_in, d_out, n_obs, period1, period2);
    CUDA_CHECK(cudaPeekAtLastError());
  }
  // If no difference and the pointers are different, copy in to out
  else if (d + D == 0 && d_in != d_out) {
    MLCommon::copy(d_out, d_in, n_obs * batch_size, stream);
  }
  // Other cases: no difference and the pointers are the same, nothing to do

  // Remove trend in-place
  if (intercept) {
    MLCommon::LinAlg::matrixVectorOp(
      d_out, d_out, d_mu, batch_size, n_obs - d - D * s, false, true,
      [] __device__(DataT a, DataT b) { return a - b; }, stream);
  }
}

/**
 * @brief Helper function that will read in src0 if the given index is
 *        negative, src1 otherwise.
 * @note  This is useful when one array is the logical continuation of
 *        another and the index is expressed relatively to the second array.
 */
template <typename DataT>
inline __device__ DataT _select_read(const DataT* src0, int size0,
                                     const DataT* src1, int idx) {
  return idx < 0 ? src0[size0 + idx] : src1[idx];
}

/**
 * @brief Kernel to undifference the data with up to two levels of simple
 *        and/or seasonal differencing.
 * @note  One thread per series.
 */
template <bool double_diff, typename DataT>
__global__ void _undiff_kernel(DataT* d_fc, const DataT* d_in, int num_steps,
                               int batch_size, int in_ld, int n_in, int s0,
                               int s1 = 0) {
  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (bid < batch_size) {
    DataT* b_fc = d_fc + bid * num_steps;
    const DataT* b_in = d_in + bid * in_ld;
    for (int i = 0; i < num_steps; i++) {
      if (!double_diff) {  // One simple or seasonal difference
        b_fc[i] += _select_read(b_in, n_in, b_fc, i - s0);
      } else {  // Two differences (simple, seasonal or both)
        DataT fc_acc = -_select_read(b_in, n_in, b_fc, i - s0 - s1);
        fc_acc += _select_read(b_in, n_in, b_fc, i - s0);
        fc_acc += _select_read(b_in, n_in, b_fc, i - s1);
        b_fc[i] += fc_acc;
      }
    }
  }
}

/**
 * @brief Finalizes a forecast by adding the trend and/or undifferencing
 *
 * @note: It is assumed that d + D <= 2. This is enforced on the Python side
 *
 * @tparam        DataT       Scalar type
 * @param[in|out] d_fc        Forecast. Shape (num_steps, batch_size) (device)
 * @param[in]     d_in        Original data. Shape (n_in, batch_size) (device)
 * @param[in]     num_steps   Number of steps forecasted
 * @param[in]     batch_size  Number of series per batch
 * @param[in]     in_ld       Leading dimension of d_in
 * @param[in]     n_in        Number of observations/predictions in d_in
 * @param[in]     d           Order of simple differences (0, 1 or 2)
 * @param[in]     D           Order of seasonal differences (0, 1 or 2)
 * @param[in]     s           Seasonal period if D > 0
 * @param[in]     stream      CUDA stream
 * @param[in]     intercept   Whether the model fits an intercept
 * @param[in]     d_mu        Mu array if intercept > 0
 *                            Shape (batch_size,) (device)
 */
template <typename DataT>
void finalize_forecast(DataT* d_fc, const DataT* d_in, int num_steps,
                       int batch_size, int in_ld, int n_in, int d, int D, int s,
                       cudaStream_t stream, int intercept = 0,
                       const DataT* d_mu = nullptr) {
  // Add the trend in-place
  if (intercept) {
    MLCommon::LinAlg::matrixVectorOp(
      d_fc, d_fc, d_mu, batch_size, num_steps, false, true,
      [] __device__(DataT a, DataT b) { return a + b; }, stream);
  }

  // Undifference
  constexpr int TPB = 64;  // One thread per series -> avoid big blocks
  if (d + D == 1) {
    _undiff_kernel<false>
      <<<MLCommon::ceildiv<int>(batch_size, TPB), TPB, 0, stream>>>(
        d_fc, d_in, num_steps, batch_size, in_ld, n_in, d ? 1 : s);
    CUDA_CHECK(cudaPeekAtLastError());
  } else if (d + D == 2) {
    _undiff_kernel<true>
      <<<MLCommon::ceildiv<int>(batch_size, TPB), TPB, 0, stream>>>(
        d_fc, d_in, num_steps, batch_size, in_ld, n_in, d ? 1 : s,
        d == 2 ? 1 : s);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

/* AR and MA parameters have to be within a "triangle" region (i.e., subject to
 * an inequality) for the inverse transform to not return 'NaN' due to the
 * logarithm within the inverse. This function ensures that inequality is
 * satisfied for all parameters.
 */
template <typename DataT>
void fix_ar_ma_invparams(const DataT* d_old_params, DataT* d_new_params,
                         int batch_size, int pq, cudaStream_t stream,
                         bool isAr = true) {
  CUDA_CHECK(cudaMemcpyAsync(d_new_params, d_old_params,
                             batch_size * pq * sizeof(DataT),
                             cudaMemcpyDeviceToDevice, stream));
  int n = pq;

  // The parameter must be within a "triangle" region. If not, we bring the
  // parameter inside by 1%.
  DataT eps = 0.99;
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size,
    [=] __device__(int ib) {
      for (int i = 0; i < n; i++) {
        DataT sum = 0.0;
        for (int j = 0; j < i; j++) {
          sum += d_new_params[n - j - 1 + ib * n];
        }
        // AR is minus
        if (isAr) {
          // param < 1-sum(param)
          d_new_params[n - i - 1 + ib * n] =
            fmin((1 - sum) * eps, d_new_params[n - i - 1 + ib * n]);
          // param > -(1-sum(param))
          d_new_params[n - i - 1 + ib * n] =
            fmax(-(1 - sum) * eps, d_new_params[n - i - 1 + ib * n]);
        } else {
          // MA is plus
          // param < 1+sum(param)
          d_new_params[n - i - 1 + ib * n] =
            fmin((1 + sum) * eps, d_new_params[n - i - 1 + ib * n]);
          // param > -(1+sum(param))
          d_new_params[n - i - 1 + ib * n] =
            fmax(-(1 + sum) * eps, d_new_params[n - i - 1 + ib * n]);
        }
      }
    });
}

/**
 * Auxiliary function of batched_jones_transform to remove redundancy.
 * Applies the transform to the given batched parameters.
 */
template <typename DataT>
void _transform_helper(const DataT* d_param, DataT* d_Tparam, int k,
                       int batch_size, bool isInv, bool isAr,
                       std::shared_ptr<deviceAllocator> allocator,
                       cudaStream_t stream) {
  // inverse transform will produce NaN if parameters are outside of a
  // "triangle" region
  DataT* d_param_fixed =
    (DataT*)allocator->allocate(sizeof(DataT) * batch_size * k, stream);
  if (isInv) {
    fix_ar_ma_invparams(d_param, d_param_fixed, batch_size, k, stream, isAr);
  } else {
    CUDA_CHECK(cudaMemcpyAsync(d_param_fixed, d_param,
                               sizeof(DataT) * batch_size * k,
                               cudaMemcpyDeviceToDevice, stream));
  }
  MLCommon::TimeSeries::jones_transform(d_param_fixed, batch_size, k, d_Tparam,
                                        isAr, isInv, allocator, stream);

  allocator->deallocate(d_param_fixed, sizeof(DataT) * batch_size * k, stream);
}

/**
 * Convenience function for batched "jones transform" used in ARIMA to ensure
 * certain properties of the AR and MA parameters
 *
 * @tparam     DataT      Scalar type
 * @param[in]  order      ARIMA hyper-parameters
 * @param[in]  batch_size Number of time series analyzed.
 * @param[in]  isInv      Do the inverse transform?
 * @param[in]  params     ARIMA parameters (device)
 * @param[in]  Tparams    Transformed ARIMA parameters (device)
 * @param[in]  allocator  Device memory allocator
 * @param[in]  stream     CUDA stream
 */
template <typename DataT>
void batched_jones_transform(const ML::ARIMAOrder& order, int batch_size,
                             bool isInv, const ML::ARIMAParams<DataT>& params,
                             const ML::ARIMAParams<DataT>& Tparams,
                             std::shared_ptr<deviceAllocator> allocator,
                             cudaStream_t stream) {
  if (order.p)
    _transform_helper(params.ar, Tparams.ar, order.p, batch_size, isInv, true,
                      allocator, stream);
  if (order.q)
    _transform_helper(params.ma, Tparams.ma, order.q, batch_size, isInv, false,
                      allocator, stream);
  if (order.P)
    _transform_helper(params.sar, Tparams.sar, order.P, batch_size, isInv, true,
                      allocator, stream);
  if (order.Q)
    _transform_helper(params.sma, Tparams.sma, order.Q, batch_size, isInv,
                      false, allocator, stream);
}

}  // namespace TimeSeries
}  // namespace MLCommon
