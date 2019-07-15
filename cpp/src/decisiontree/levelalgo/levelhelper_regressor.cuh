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
#include "levelkernel_regressor.cuh"
template <typename T, typename F>
void initial_metric_regression(T *labels, unsigned int *sample_cnt,
                               const int nrows, T &mean, T &initial_metric,
                               std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  int threads = 128;
  int blocks = MLCommon::ceildiv(nrows, threads);
  if (blocks > 65536) blocks = 65536;
  pred_kernel_level<<<blocks, threads, 0, tempmem->stream>>>(
    labels, sample_cnt, nrows, tempmem->d_predout->data(),
    tempmem->d_count->data());
  CUDA_CHECK(cudaGetLastError());
  mse_kernel_level<T, F> <<< blocks, threads, 0,
    tempmem->stream >>> (labels, sample_cnt, nrows, tempmem->d_predout->data(),
                         tempmem->d_count->data(), tempmem->d_mseout->data());
  CUDA_CHECK(cudaGetLastError());
  MLCommon::updateHost(tempmem->h_count->data(), tempmem->d_count->data(), 1,
                       tempmem->stream);
  MLCommon::updateHost(tempmem->h_predout->data(), tempmem->d_predout->data(),
                       1, tempmem->stream);
  MLCommon::updateHost(tempmem->h_mseout->data(), tempmem->d_mseout->data(), 1,
                       tempmem->stream);
  CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  mean = tempmem->h_predout->data()[0] / tempmem->h_count->data()[0];
  initial_metric = tempmem->h_mseout->data()[0] / tempmem->h_count->data()[0];
}
