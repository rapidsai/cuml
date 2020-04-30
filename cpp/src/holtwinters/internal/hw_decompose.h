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
#include <common/cudart_utils.h>
#include "hw_utils.h"

// optimize, maybe im2col ?
// https://github.com/rapidsai/cuml/issues/891
template <typename Dtype>
__global__ void conv1d_kernel(const Dtype *input, int batch_size,
                              const Dtype *filter, int filter_size,
                              Dtype *output, int output_size) {
  const int tid = GET_TID;
  if (tid < batch_size) {
    for (int o = 0; o < output_size; ++o) {
      Dtype out = 0.;
      for (int i = 0; i < filter_size; ++i)
        out += filter[i] * input[tid + (i + o) * batch_size];
      output[tid + o * batch_size] = out;
    }
  }
}

template <typename Dtype>
void conv1d(const ML::cumlHandle_impl &handle, const Dtype *input,
            int batch_size, const Dtype *filter, int filter_size, Dtype *output,
            int output_size) {
  int total_threads = batch_size;
  conv1d_kernel<Dtype>
    <<<GET_NUM_BLOCKS(total_threads), GET_THREADS_PER_BLOCK(total_threads), 0,
       handle.getStream()>>>(input, batch_size, filter, filter_size, output,
                             output_size);
}

//https://github.com/rapidsai/cuml/issues/891
template <typename Dtype>
__global__ void season_mean_kernel(const Dtype *season, int len, int batch_size,
                                   Dtype *start_season, int frequency,
                                   int half_filter_size, bool ADDITIVE_KERNEL) {
  int tid = GET_TID;
  if (tid < batch_size) {
    Dtype mean = 0.0;
    for (int i = 0; i < frequency; ++i) {
      Dtype period_mean = 0.;
      int c = 0;
      for (int k = i; k < len; k = k + frequency) {
        period_mean += season[k * batch_size + tid];
        c++;
      }
      int count = 1 + ((len - i - 1) / frequency);
      period_mean /= count;
      int ss_idx = (i + half_filter_size) % frequency;
      start_season[ss_idx * batch_size + tid] = period_mean;
      mean += period_mean;
    }
    mean /= frequency;
    for (int i = 0; i < frequency; ++i) {
      if (ADDITIVE_KERNEL)
        start_season[i * batch_size + tid] -= mean;
      else  // MULTIPLOCATIVE
        start_season[i * batch_size + tid] /= mean;
    }
  }
}

template <typename Dtype>
void season_mean(const ML::cumlHandle_impl &handle, const Dtype *season,
                 int len, int batch_size, Dtype *start_season, int frequency,
                 int half_filter_size, ML::SeasonalType seasonal) {
  cudaStream_t stream = handle.getStream();
  bool is_additive = seasonal == ML::SeasonalType::ADDITIVE;
  season_mean_kernel<Dtype>
    <<<GET_NUM_BLOCKS(batch_size), GET_THREADS_PER_BLOCK(batch_size), 0,
       stream>>>(season, len, batch_size, start_season, frequency,
                 half_filter_size, is_additive);
}

template <typename Dtype>
__global__ void RinvKernel(const Dtype *A, Dtype *Rinv, int trend_len) {
  // Inverse of R (2x2 upper triangular matrix)
  int tid = GET_TID;
  if (tid == 0) {
    Dtype a = A[0], b = A[trend_len], d = A[trend_len + 1];
    Dtype factor = 1. / (a * d);
    Rinv[0] = factor * d;
    Rinv[1] = 0.;
    Rinv[2] = -factor * b;
    Rinv[3] = factor * a;
  }
}

template <typename Dtype>
__global__ void batched_ls_solver_kernel(const Dtype *B, const Dtype *rq,
                                         int batch_size, int len, Dtype *level,
                                         Dtype *trend) {
  int tid = GET_TID;
  if (tid < batch_size) {
    Dtype level_ = 0., trend_ = 0.;
    for (int i = 0; i < len; ++i) {
      Dtype b = B[tid + i * batch_size];
      level_ += rq[2 * i] * b;
      trend_ += rq[2 * i + 1] * b;
    }
    level[tid] = level_;
    trend[tid] = trend_;
  }
}

template <typename Dtype>
void batched_ls(const ML::cumlHandle_impl &handle, const Dtype *data,
                int trend_len, int batch_size, Dtype *level, Dtype *trend) {
  cudaStream_t stream = handle.getStream();
  cublasHandle_t cublas_h = handle.getCublasHandle();
  cusolverDnHandle_t cusolver_h = handle.getcusolverDnHandle();
  std::shared_ptr<MLCommon::deviceAllocator> dev_allocator =
    handle.getDeviceAllocator();

  const Dtype one = (Dtype)1.;
  const Dtype zero = (Dtype)0.;
  int geqrf_buffer;
  int orgqr_buffer;
  int lwork_size;

  // Allocate memory
  std::vector<Dtype> A_h(2 * trend_len);

  MLCommon::device_buffer<Dtype> A_d(dev_allocator, stream, 2 * trend_len);
  MLCommon::device_buffer<Dtype> tau_d(dev_allocator, stream, 2);
  MLCommon::device_buffer<Dtype> Rinv_d(dev_allocator, stream, 4);
  MLCommon::device_buffer<Dtype> R1Qt_d(dev_allocator, stream, 2 * trend_len);
  MLCommon::device_buffer<int> dev_info_d(dev_allocator, stream, 1);

  // Prepare A
  for (int i = 0; i < trend_len; ++i) {
    A_h[i] = (Dtype)1.;
    A_h[trend_len + i] = (Dtype)(i + 1);
  }
  MLCommon::updateDevice(A_d.data(), A_h.data(), 2 * trend_len, stream);

  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDngeqrf_bufferSize<Dtype>(
    cusolver_h, trend_len, 2, A_d.data(), 2, &geqrf_buffer));

  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDnorgqr_bufferSize<Dtype>(
    cusolver_h, trend_len, 2, 2, A_d.data(), 2, tau_d.data(), &orgqr_buffer));

  lwork_size = geqrf_buffer > orgqr_buffer ? geqrf_buffer : orgqr_buffer;
  MLCommon::device_buffer<Dtype> lwork_d(dev_allocator, stream, lwork_size);

  // QR decomposition of A
  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDngeqrf<Dtype>(
    cusolver_h, trend_len, 2, A_d.data(), trend_len, tau_d.data(),
    lwork_d.data(), lwork_size, dev_info_d.data(), stream));

  // Single thread kenrel to inverse R
  RinvKernel<Dtype><<<1, 1, 0, stream>>>(A_d.data(), Rinv_d.data(), trend_len);

  // R1QT = inv(R)*transpose(Q)
  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDnorgqr<Dtype>(
    cusolver_h, trend_len, 2, 2, A_d.data(), trend_len, tau_d.data(),
    lwork_d.data(), lwork_size, dev_info_d.data(), stream));

  CUBLAS_CHECK(MLCommon::LinAlg::cublasgemm<Dtype>(
    cublas_h, CUBLAS_OP_N, CUBLAS_OP_T, 2, trend_len, 2, &one, Rinv_d.data(), 2,
    A_d.data(), trend_len, &zero, R1Qt_d.data(), 2, stream));

  batched_ls_solver_kernel<Dtype>
    <<<GET_NUM_BLOCKS(batch_size), GET_THREADS_PER_BLOCK(batch_size), 0,
       stream>>>(data, R1Qt_d.data(), batch_size, trend_len, level, trend);
}

template <typename Dtype>
void stl_decomposition_gpu(const ML::cumlHandle_impl &handle, const Dtype *ts,
                           int n, int batch_size, int frequency,
                           int start_periods, Dtype *start_level,
                           Dtype *start_trend, Dtype *start_season,
                           ML::SeasonalType seasonal) {
  cudaStream_t stream = handle.getStream();
  cublasHandle_t cublas_h = handle.getCublasHandle();
  std::shared_ptr<MLCommon::deviceAllocator> dev_allocator =
    handle.getDeviceAllocator();

  const int end = start_periods * frequency;
  const int filter_size = (frequency / 2) * 2 + 1;
  const int trend_len = end - filter_size + 1;

  // Set filter
  std::vector<Dtype> filter_h(filter_size, 1. / frequency);
  if (frequency % 2 == 0) {
    filter_h.front() /= 2;
    filter_h.back() /= 2;
  }

  MLCommon::device_buffer<Dtype> filter_d(dev_allocator, stream, filter_size);
  MLCommon::updateDevice(filter_d.data(), filter_h.data(), filter_size, stream);

  // Set Trend
  MLCommon::device_buffer<Dtype> trend_d(dev_allocator, stream,
                                         batch_size * trend_len);
  conv1d<Dtype>(handle, ts, batch_size, filter_d.data(), filter_size,
                trend_d.data(), trend_len);

  MLCommon::device_buffer<Dtype> season_d(dev_allocator, stream,
                                          batch_size * trend_len);

  const int ts_offset = (filter_size / 2) * batch_size;
  if (seasonal == ML::SeasonalType::ADDITIVE) {
    const Dtype one = 1.;
    const Dtype minus_one = -1.;
    CUBLAS_CHECK(MLCommon::LinAlg::cublasgeam<Dtype>(
      cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, trend_len, batch_size, &one,
      ts + ts_offset, trend_len, &minus_one, trend_d.data(), trend_len,
      season_d.data(), trend_len, stream));
  } else {
    MLCommon::device_buffer<Dtype> aligned_ts(dev_allocator, stream,
                                              batch_size * trend_len);
    MLCommon::copy(aligned_ts.data(), ts + ts_offset, batch_size * trend_len,
                   stream);
    MLCommon::LinAlg::eltwiseDivide<Dtype>(season_d.data(), aligned_ts.data(),
                                           trend_d.data(),
                                           trend_len * batch_size, stream);
  }

  season_mean(handle, season_d.data(), trend_len, batch_size, start_season,
              frequency, filter_size / 2, seasonal);

  batched_ls(handle, trend_d.data(), trend_len, batch_size, start_level,
             start_trend);
}
