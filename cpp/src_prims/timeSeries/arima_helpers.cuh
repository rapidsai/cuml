/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "jones_transform.cuh"

#include <cuml/tsa/arima_common.h>

#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

#include <linalg/batched/matrix.cuh>

// Private helper functions and kernels in the anonymous namespace
namespace {

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
HDI DataT _param_to_poly(const DataT* param, int lags, int idx)
{
  if (idx > lags) {
    return 0.0;
  } else if (idx) {
    return isAr ? -param[idx - 1] : param[idx - 1];
  } else
    return 1.0;
}

/**
 * @brief Helper function that will read in src0 if the given index is
 *        negative, src1 otherwise.
 * @note  This is useful when one array is the logical continuation of
 *        another and the index is expressed relatively to the second array.
 *
 * @param[in] src0  Data comes from here if the index is negative
 * @param[in] size0 Size of src0
 * @param[in] src1  Data comes from here if the index is positive
 * @param[in] idx   Index, relative to the start of the second array src1
 * @return Data read from src0 or src1 according to the index
 */
template <typename DataT>
DI DataT _select_read(const DataT* src0, int size0, const DataT* src1, int idx)
{
  return idx < 0 ? src0[size0 + idx] : src1[idx];
}

/**
 * @brief Prepare future data with a simple or seasonal difference
 *
 * @param[in]  in_past  Input (past). Shape (n_past, batch_size) (device)
 * @param[in]  in_fut   Input (future). Shape (n_fut, batch_size) (device)
 * @param[out] out      Output. Shape (n_fut, batch_size) (device)
 * @param[in]  n_past   Number of past observations per series
 * @param[in]  n_fut    Number of future observations per series
 * @param[in]  period   Differencing period (1 or s)
 * @param[in]  stream   CUDA stream
 */
template <typename T>
CUML_KERNEL void _future_diff_kernel(
  const T* in_past, const T* in_fut, T* out, int n_past, int n_fut, int period = 1)
{
  const T* b_in_past = in_past + n_past * blockIdx.x;
  const T* b_in_fut  = in_fut + n_fut * blockIdx.x;
  T* b_out           = out + n_fut * blockIdx.x;

  for (int i = threadIdx.x; i < n_fut; i += blockDim.x) {
    b_out[i] = b_in_fut[i] - _select_read(b_in_past, n_past, b_in_fut, i - period);
  }
}

/**
 * @brief Prepare future data with two simple and/or seasonal differences
 *
 * @param[in]  in_past  Input (past). Shape (n_past, batch_size) (device)
 * @param[in]  in_fut   Input (future). Shape (n_fut, batch_size) (device)
 * @param[out] out      Output. Shape (n_fut, batch_size) (device)
 * @param[in]  n_past   Number of past observations per series
 * @param[in]  n_fut    Number of future observations per series
 * @param[in]  period1  First differencing period (1 or s)
 * @param[in]  period2  Second differencing period (1 or s)
 * @param[in]  stream   CUDA stream
 */
template <typename T>
CUML_KERNEL void _future_second_diff_kernel(const T* in_past,
                                            const T* in_fut,
                                            T* out,
                                            int n_past,
                                            int n_fut,
                                            int period1 = 1,
                                            int period2 = 1)
{
  const T* b_in_past = in_past + n_past * blockIdx.x;
  const T* b_in_fut  = in_fut + n_fut * blockIdx.x;
  T* b_out           = out + n_fut * blockIdx.x;

  for (int i = threadIdx.x; i < n_fut; i += blockDim.x) {
    b_out[i] = b_in_fut[i] - _select_read(b_in_past, n_past, b_in_fut, i - period1) -
               _select_read(b_in_past, n_past, b_in_fut, i - period2) +
               _select_read(b_in_past, n_past, b_in_fut, i - period1 - period2);
  }
}

/**
 * @brief Kernel to undifference the data with up to two levels of simple
 *        and/or seasonal differencing.
 * @note  One thread per series.
 *
 * @tparam       double_diff true for two differences, false for one
 * @tparam       DataT       Data type
 * @param[inout] d_fc        Forecasts, modified in-place
 * @param[in]    d_in        Past observations
 * @param[in]    num_steps   Number of forecast steps
 * @param[in]    batch_size  Batch size
 * @param[in]    in_ld       Leading dimension of d_in
 * @param[in]    n_in        Number of past observations
 * @param[in]    s0          1st differencing period
 * @param[in]    s1          2nd differencing period if relevant
 */
template <bool double_diff, typename DataT>
CUML_KERNEL void _undiff_kernel(DataT* d_fc,
                                const DataT* d_in,
                                int num_steps,
                                int batch_size,
                                int in_ld,
                                int n_in,
                                int s0,
                                int s1 = 0)
{
  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (bid < batch_size) {
    DataT* b_fc       = d_fc + bid * num_steps;
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

}  // namespace

namespace MLCommon {
namespace TimeSeries {

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
HDI DataT reduced_polynomial(
  int bid, const DataT* param, int lags, const DataT* sparam, int slags, int s, int idx)
{
  int idx1    = s ? idx / s : 0;
  int idx0    = idx - s * idx1;
  DataT coef0 = _param_to_poly<isAr>(param + bid * lags, lags, idx0);
  DataT coef1 = _param_to_poly<isAr>(sparam + bid * slags, slags, idx1);
  return isAr ? -coef0 * coef1 : coef0 * coef1;
}

/**
 * @brief Prepare data by differencing if needed (simple and/or seasonal)
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
 */
template <typename DataT>
void prepare_data(DataT* d_out,
                  const DataT* d_in,
                  int batch_size,
                  int n_obs,
                  int d,
                  int D,
                  int s,
                  cudaStream_t stream)
{
  // Only one difference (simple or seasonal)
  if (d + D == 1) {
    int period = d ? 1 : s;
    int tpb    = (n_obs - period) > 512 ? 256 : 128;  // quick heuristics
    MLCommon::LinAlg::Batched::batched_diff_kernel<<<batch_size, tpb, 0, stream>>>(
      d_in, d_out, n_obs, period);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  // Two differences (simple or seasonal or both)
  else if (d + D == 2) {
    int period1 = d ? 1 : s;
    int period2 = d == 2 ? 1 : s;
    int tpb     = (n_obs - period1 - period2) > 512 ? 256 : 128;
    MLCommon::LinAlg::Batched::batched_second_diff_kernel<<<batch_size, tpb, 0, stream>>>(
      d_in, d_out, n_obs, period1, period2);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  // If no difference and the pointers are different, copy in to out
  else if (d + D == 0 && d_in != d_out) {
    raft::copy(d_out, d_in, n_obs * batch_size, stream);
  }
  // Other cases: no difference and the pointers are the same, nothing to do
}

/**
 * @brief Prepare future data by differencing if needed (simple and/or seasonal)
 *
 * This is a variant of prepare_data that produces an output of the same dimension
 * as the input, using an other array of past data for the observations at the start
 *
 * @note: It is assumed that d + D <= 2. This is enforced on the Python side
 *
 * @param[out] d_out       Output. Shape (n_fut, batch_size) (device)
 * @param[in]  d_in_past   Input (past). Shape (n_past, batch_size) (device)
 * @param[in]  d_in_fut    Input (future). Shape (n_fut, batch_size) (device)
 * @param[in]  batch_size  Number of series per batch
 * @param[in]  n_past      Number of past observations per series
 * @param[in]  n_fut       Number of future observations per series
 * @param[in]  d           Order of simple differences (0, 1 or 2)
 * @param[in]  D           Order of seasonal differences (0, 1 or 2)
 * @param[in]  s           Seasonal period if D > 0
 * @param[in]  stream      CUDA stream
 */
template <typename DataT>
void prepare_future_data(DataT* d_out,
                         const DataT* d_in_past,
                         const DataT* d_in_fut,
                         int batch_size,
                         int n_past,
                         int n_fut,
                         int d,
                         int D,
                         int s,
                         cudaStream_t stream)
{
  // Only one difference (simple or seasonal)
  if (d + D == 1) {
    int period = d ? 1 : s;
    int tpb    = n_fut > 128 ? 64 : 32;  // quick heuristics
    _future_diff_kernel<<<batch_size, tpb, 0, stream>>>(
      d_in_past, d_in_fut, d_out, n_past, n_fut, period);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  // Two differences (simple or seasonal or both)
  else if (d + D == 2) {
    int period1 = d ? 1 : s;
    int period2 = d == 2 ? 1 : s;
    int tpb     = n_fut > 128 ? 64 : 32;
    _future_second_diff_kernel<<<batch_size, tpb, 0, stream>>>(
      d_in_past, d_in_fut, d_out, n_past, n_fut, period1, period2);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  // If no difference and the pointers are different, copy in to out
  else if (d + D == 0 && d_in_fut != d_out) {
    raft::copy(d_out, d_in_fut, n_fut * batch_size, stream);
  }
  // Other cases: no difference and the pointers are the same, nothing to do
}

/**
 * @brief Finalizes a forecast by undifferencing.
 *
 * This is used when doing "simple differencing" for integrated models (d > 0 or D > 0), i.e the
 * series are differenced prior to running the Kalman filter. Forecasts output by the Kalman filter
 * are then for the differenced series and we need to couple this with past observations to compute
 * forecasts for the non-differenced series. This is not needed when differencing is handled by the
 * Kalman filter.
 *
 * @note: It is assumed that d + D <= 2. This is enforced on the Python side
 *
 * @tparam       DataT       Scalar type
 * @param[inout] d_fc        Forecast. Shape (num_steps, batch_size) (device)
 * @param[in]    d_in        Original data. Shape (n_in, batch_size) (device)
 * @param[in]    num_steps   Number of steps forecasted
 * @param[in]    batch_size  Number of series per batch
 * @param[in]    in_ld       Leading dimension of d_in
 * @param[in]    n_in        Number of observations/predictions in d_in
 * @param[in]    d           Order of simple differences (0, 1 or 2)
 * @param[in]    D           Order of seasonal differences (0, 1 or 2)
 * @param[in]    s           Seasonal period if D > 0
 * @param[in]    stream      CUDA stream
 */
template <typename DataT>
void finalize_forecast(DataT* d_fc,
                       const DataT* d_in,
                       int num_steps,
                       int batch_size,
                       int in_ld,
                       int n_in,
                       int d,
                       int D,
                       int s,
                       cudaStream_t stream)
{
  // Undifference
  constexpr int TPB = 64;  // One thread per series -> avoid big blocks
  if (d + D == 1) {
    _undiff_kernel<false><<<raft::ceildiv<int>(batch_size, TPB), TPB, 0, stream>>>(
      d_fc, d_in, num_steps, batch_size, in_ld, n_in, d ? 1 : s);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else if (d + D == 2) {
    _undiff_kernel<true><<<raft::ceildiv<int>(batch_size, TPB), TPB, 0, stream>>>(
      d_fc, d_in, num_steps, batch_size, in_ld, n_in, d ? 1 : s, d == 2 ? 1 : s);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
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
 * @param[in]  stream     CUDA stream
 */
template <typename DataT>
void batched_jones_transform(const ML::ARIMAOrder& order,
                             int batch_size,
                             bool isInv,
                             const ML::ARIMAParams<DataT>& params,
                             const ML::ARIMAParams<DataT>& Tparams,
                             cudaStream_t stream)
{
  if (order.p) jones_transform(params.ar, batch_size, order.p, Tparams.ar, true, isInv, stream);
  if (order.q) jones_transform(params.ma, batch_size, order.q, Tparams.ma, false, isInv, stream);
  if (order.P) jones_transform(params.sar, batch_size, order.P, Tparams.sar, true, isInv, stream);
  if (order.Q) jones_transform(params.sma, batch_size, order.Q, Tparams.sma, false, isInv, stream);

  // Constrain sigma2 to be strictly positive
  constexpr DataT min_sigma2 = 1e-6;
  raft::linalg::unaryOp<DataT>(
    Tparams.sigma2,
    params.sigma2,
    batch_size,
    [=] __device__(DataT input) { return max(input, min_sigma2); },
    stream);
}

}  // namespace TimeSeries
}  // namespace MLCommon
