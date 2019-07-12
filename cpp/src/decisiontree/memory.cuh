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
#include <thrust/extrema.h>
#include <utils.h>
#include "cub/cub.cuh"
#include "memory.h"

template <class T, class L>
TemporaryMemory<T, L>::TemporaryMemory(const ML::cumlHandle_impl& handle, int N,
                                       int Ncols, int maxstr, int n_unique,
                                       int n_bins, const int split_algo)
  : ml_handle(handle) {
  //Assign Stream from cumlHandle
  stream = ml_handle.getStream();
  NodeMemAllocator(N, Ncols, maxstr, n_unique, n_bins, split_algo);
}

template <class T, class L>
void TemporaryMemory<T, L>::NodeMemAllocator(int N, int Ncols, int maxstr,
                                             int n_unique, int n_bins,
                                             const int split_algo) {
  int n_hist_elements = n_unique * n_bins;

  h_hist = new MLCommon::host_buffer<int>(ml_handle.getHostAllocator(), stream,
                                          n_hist_elements);
  d_hist = new MLCommon::device_buffer<int>(ml_handle.getDeviceAllocator(),
                                            stream, n_hist_elements);
  nrowsleftright =
    new MLCommon::host_buffer<int>(ml_handle.getHostAllocator(), stream, 2);

  int extra_elements = Ncols;
  int quantile_elements =
    (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) ? extra_elements : 1;

  temp_data = new MLCommon::device_buffer<T>(ml_handle.getDeviceAllocator(),
                                             stream, N * Ncols);
  totalmem += n_hist_elements * sizeof(int) + N * extra_elements * sizeof(T);

  if (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) {
    h_quantile = new MLCommon::host_buffer<T>(
      ml_handle.getHostAllocator(), stream, n_bins * quantile_elements);
    d_quantile = new MLCommon::device_buffer<T>(
      ml_handle.getDeviceAllocator(), stream, n_bins * quantile_elements);
    totalmem += n_bins * extra_elements * sizeof(T);
  }

  sampledlabels =
    new MLCommon::device_buffer<L>(ml_handle.getDeviceAllocator(), stream, N);
  totalmem += N * sizeof(L);

  //Allocate Temporary for split functions
  d_num_selected_out =
    new MLCommon::device_buffer<int>(ml_handle.getDeviceAllocator(), stream, 1);
  d_flags_left = new MLCommon::device_buffer<char>(
    ml_handle.getDeviceAllocator(), stream, N);
  d_flags_right = new MLCommon::device_buffer<char>(
    ml_handle.getDeviceAllocator(), stream, N);
  temprowids = new MLCommon::device_buffer<unsigned int>(
    ml_handle.getDeviceAllocator(), stream, N);
  question_value =
    new MLCommon::device_buffer<T>(ml_handle.getDeviceAllocator(), stream, 1);

  cub::DeviceSelect::Flagged(d_split_temp_storage, split_temp_storage_bytes,
                             temprowids->data(), d_flags_left->data(),
                             temprowids->data(), d_num_selected_out->data(), N,
                             stream);
  d_split_temp_storage = new MLCommon::device_buffer<char>(
    ml_handle.getDeviceAllocator(), stream, split_temp_storage_bytes);

  totalmem += split_temp_storage_bytes + (N + 1) * sizeof(int) +
              2 * N * sizeof(char) + sizeof(T);

  h_histout = new MLCommon::host_buffer<int>(ml_handle.getHostAllocator(),
                                             stream, n_hist_elements * Ncols);
  int mse_elements = Ncols * n_bins;
  h_mseout = new MLCommon::host_buffer<T>(ml_handle.getHostAllocator(), stream,
                                          2 * mse_elements);
  h_predout = new MLCommon::host_buffer<T>(ml_handle.getHostAllocator(), stream,
                                           mse_elements);

  d_globalminmax = new MLCommon::device_buffer<T>(
    ml_handle.getDeviceAllocator(), stream, Ncols * 2);
  d_histout = new MLCommon::device_buffer<int>(ml_handle.getDeviceAllocator(),
                                               stream, n_hist_elements * Ncols);
  d_mseout = new MLCommon::device_buffer<T>(ml_handle.getDeviceAllocator(),
                                            stream, 2 * mse_elements);
  d_predout = new MLCommon::device_buffer<T>(ml_handle.getDeviceAllocator(),
                                             stream, mse_elements);

  d_colids = new MLCommon::device_buffer<unsigned int>(
    ml_handle.getDeviceAllocator(), stream, Ncols);
  // memory of d_histout + d_colids + d_globalminmax + (d_mseout + d_predout)
  totalmem += (n_hist_elements * sizeof(int) + sizeof(unsigned int) +
               2 * sizeof(T) + 3 * n_bins * sizeof(T)) *
              Ncols;

  //this->print_info();
}

template <class T, class L>
void TemporaryMemory<T, L>::print_info() {
  std::cout << " Total temporary memory usage--> "
            << ((double)totalmem / (1024 * 1024)) << "  MB" << std::endl;
}

template <class T, class L>
TemporaryMemory<T, L>::~TemporaryMemory() {
  NodeMemCleaner();
}
template <class T, class L>
void TemporaryMemory<T, L>::NodeMemCleaner() {
  h_hist->release(stream);
  d_hist->release(stream);
  nrowsleftright->release(stream);
  temp_data->release(stream);

  delete h_hist;
  delete d_hist;
  delete temp_data;

  if (d_quantile != nullptr) {
    d_quantile->release(stream);
    h_quantile->release(stream);
    delete h_quantile;
    delete d_quantile;
  }

  sampledlabels->release(stream);
  d_split_temp_storage->release(stream);
  d_num_selected_out->release(stream);
  d_flags_left->release(stream);
  d_flags_right->release(stream);
  temprowids->release(stream);
  question_value->release(stream);
  h_histout->release(stream);
  h_mseout->release(stream);
  h_predout->release(stream);

  delete sampledlabels;
  delete d_split_temp_storage;
  delete d_num_selected_out;
  delete d_flags_left;
  delete d_flags_right;
  delete temprowids;
  delete question_value;
  delete h_histout;
  delete h_mseout;
  delete h_predout;

  d_globalminmax->release(stream);
  d_histout->release(stream);
  d_mseout->release(stream);
  d_predout->release(stream);
  d_colids->release(stream);

  delete d_globalminmax;
  delete d_histout;
  delete d_mseout;
  delete d_predout;
  delete d_colids;
}
