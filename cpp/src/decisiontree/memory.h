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
#include <utils.h>
#include <common/device_buffer.hpp>
#include <common/host_buffer.hpp>
#include "common/cumlHandle.hpp"

template <class T, class L>
struct TemporaryMemory {
  // Labels after boostrapping
  MLCommon::device_buffer<L> *sampledlabels = nullptr;

  // Used for gini histograms (root tree node)
  MLCommon::device_buffer<int> *d_hist = nullptr;
  MLCommon::host_buffer<int> *h_hist = nullptr;

  //Host/Device histograms and device minmaxs
  MLCommon::device_buffer<T> *d_globalminmax = nullptr;
  MLCommon::device_buffer<int> *d_histout = nullptr;
  MLCommon::device_buffer<unsigned int> *d_colids = nullptr;
  MLCommon::host_buffer<int> *h_histout = nullptr;
  MLCommon::device_buffer<T> *d_mseout = nullptr;
  MLCommon::device_buffer<T> *d_predout = nullptr;
  MLCommon::host_buffer<T> *h_mseout = nullptr;
  MLCommon::host_buffer<T> *h_predout = nullptr;

  //Below pointers are shared for split functions
  MLCommon::device_buffer<char> *d_flags_left = nullptr;
  MLCommon::device_buffer<char> *d_flags_right = nullptr;
  MLCommon::host_buffer<int> *nrowsleftright = nullptr;
  MLCommon::device_buffer<char> *d_split_temp_storage = nullptr;
  size_t split_temp_storage_bytes = 0;

  MLCommon::device_buffer<int> *d_num_selected_out = nullptr;
  MLCommon::device_buffer<unsigned int> *temprowids = nullptr;
  MLCommon::device_buffer<T> *question_value = nullptr;
  MLCommon::device_buffer<T> *temp_data = nullptr;

  //Total temp mem
  size_t totalmem = 0;

  //CUDA stream
  cudaStream_t stream;

  //For quantiles
  MLCommon::device_buffer<T> *d_quantile = nullptr;
  MLCommon::host_buffer<T> *h_quantile = nullptr;

  const ML::cumlHandle_impl &ml_handle;

  TemporaryMemory(const ML::cumlHandle_impl &handle, int N, int Ncols,
                  int maxstr, int n_unique, int n_bins, const int split_algo);
  void NodeMemAllocator(int N, int Ncols, int maxstr, int n_unique, int n_bins,
                        const int split_algo);
  void LevelMemAllocator(int N, int Ncols, int maxstr, int n_unique, int n_bins,
                         const int split_algo);
  void print_info();
  ~TemporaryMemory();
  void NodeMemCleaner();
  void LevelMemCleaner();
};
