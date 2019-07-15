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
  mse_kernel_level<T, F><<<blocks, threads, 0, tempmem->stream>>>(
    labels, sample_cnt, nrows, tempmem->d_predout->data(),
    tempmem->d_count->data(), tempmem->d_mseout->data());
  CUDA_CHECK(cudaGetLastError());
  MLCommon::updateHost(tempmem->h_count->data(), tempmem->d_count->data(), 1,
                       tempmem->stream);
  MLCommon::updateHost(tempmem->h_predout->data(), tempmem->d_predout->data(),
                       1, tempmem->stream);
  MLCommon::updateHost(tempmem->h_mseout->data(), tempmem->d_mseout->data(), 1,
                       tempmem->stream);
  MLCommon::updateDevice(tempmem->d_parent_count->data(),
                         tempmem->h_count->data(), 1, tempmem->stream);
  CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  tempmem->h_predout->data()[0] =
    tempmem->h_predout->data()[0] / tempmem->h_count->data()[0];
  mean = tempmem->h_predout->data()[0];
  MLCommon::updateDevice(tempmem->d_parent_pred->data(),
                         tempmem->h_predout->data(), 1, tempmem->stream);
  initial_metric = tempmem->h_mseout->data()[0] / tempmem->h_count->data()[0];
}

template <typename T, typename F>
void get_mse_regression(T *data, T *labels, unsigned int *flags,
                        unsigned int *sample_cnt, const int nrows,
                        const int ncols, const int nbins, const int n_nodes,
                        std::shared_ptr<TemporaryMemory<T, T>> tempmem,
                        T *d_mseout, T *d_predout) {
  size_t predcount = ncols * nbins * n_nodes;
  CUDA_CHECK(
    cudaMemsetAsync(d_mseout, 0, 2 * predcount * sizeof(T), tempmem->stream));
  CUDA_CHECK(
    cudaMemsetAsync(d_predout, 0, predcount * sizeof(T), tempmem->stream));
  CUDA_CHECK(cudaMemsetAsync(tempmem->d_count->data(), 0,
                             predcount * sizeof(unsigned int),
                             tempmem->stream));

  int node_batch_pred = min(n_nodes, tempmem->max_nodes_pred);
  int node_batch_mse = min(n_nodes, tempmem->max_nodes_mse);
  size_t shmempred = nbins * (sizeof(unsigned int) + sizeof(T)) * n_nodes;
  size_t shmemmse = shmempred + 2 * nbins * n_nodes * sizeof(T);

  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);
  if ((n_nodes == node_batch_pred) && (blocks < 65536)) {
    get_pred_kernel<<<blocks, threads, shmempred, tempmem->stream>>>(
      data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows, ncols,
      nbins, n_nodes, tempmem->d_quantile->data(), d_predout,
      tempmem->d_count->data());
  } else {
  }
  CUDA_CHECK(cudaGetLastError());
  if ((n_nodes == node_batch_mse) && (blocks < 65536)) {
    get_mse_kernel<T, F><<<blocks, threads, shmemmse, tempmem->stream>>>(
      data, labels, flags, sample_cnt, tempmem->d_colids->data(), nrows, ncols,
      nbins, n_nodes, tempmem->d_quantile->data(),
      tempmem->d_parent_pred->data(), tempmem->d_parent_count->data(),
      d_predout, tempmem->d_count->data(), d_mseout);
  } else {
  }
  CUDA_CHECK(cudaGetLastError());
}
