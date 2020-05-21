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
#include "hw_utils.cuh"

template <typename Dtype>
__global__ void holtwinters_seasonal_forecast_kernel(
  Dtype *forecast, int h, int batch_size, int frequency,
  const Dtype *level_coef, const Dtype *trend_coef, const Dtype *season_coef,
  bool additive) {
  int tid = GET_TID;
  if (tid < batch_size) {
    const Dtype level = (level_coef) ? level_coef[tid] : 0.;
    const Dtype trend = (trend_coef) ? trend_coef[tid] : 0.;
    for (int i = 0; i < h; ++i) {
      const Dtype season = season_coef[tid + (i % frequency) * batch_size];
      if (additive)
        forecast[tid + i * batch_size] = level + trend * (i + 1) + season;
      else
        forecast[tid + i * batch_size] = (level + trend * (i + 1)) * season;
    }
  }
}

template <typename Dtype>
__global__ void holtwinters_nonseasonal_forecast_kernel(
  Dtype *forecast, int h, int batch_size, const Dtype *level_coef,
  const Dtype *trend_coef) {
  int tid = GET_TID;
  if (tid < batch_size) {
    const Dtype level = (level_coef) ? level_coef[tid] : 0.;
    const Dtype trend = trend_coef[tid];
    for (int i = 0; i < h; ++i)
      forecast[tid + i * batch_size] = level + trend * (i + 1);
  }
}

template <typename Dtype>
__global__ void holtwinters_level_forecast_kernel(Dtype *forecast, int h,
                                                  int batch_size,
                                                  const Dtype *level_coef) {
  int tid = GET_TID;
  if (tid < batch_size) {
    const Dtype level = level_coef[tid];
    for (int i = 0; i < h; ++i) forecast[tid + i * batch_size] = level;
  }
}

template <typename Dtype>
void holtwinters_forecast_gpu(const ML::cumlHandle_impl &handle,
                              Dtype *forecast, int h, int batch_size,
                              int frequency, const Dtype *level_coef,
                              const Dtype *trend_coef, const Dtype *season_coef,
                              ML::SeasonalType seasonal) {
  cudaStream_t stream = handle.getStream();

  int total_blocks = GET_NUM_BLOCKS(batch_size);
  int threads_per_block = GET_THREADS_PER_BLOCK(batch_size);

  if (trend_coef == nullptr && season_coef == nullptr) {
    holtwinters_level_forecast_kernel<Dtype>
      <<<total_blocks, threads_per_block, 0, stream>>>(forecast, h, batch_size,
                                                       level_coef);
  } else if (season_coef == nullptr) {
    holtwinters_nonseasonal_forecast_kernel<Dtype>
      <<<total_blocks, threads_per_block, 0, stream>>>(forecast, h, batch_size,
                                                       level_coef, trend_coef);
  } else {
    bool is_additive = seasonal == ML::SeasonalType::ADDITIVE;
    holtwinters_seasonal_forecast_kernel<Dtype>
      <<<total_blocks, threads_per_block, 0, stream>>>(
        forecast, h, batch_size, frequency, level_coef, trend_coef, season_coef,
        is_additive);
  }
}
