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
  MLCommon::device_buffer<L> *sampledlabels;

  // Used for gini histograms (root tree node)
  MLCommon::device_buffer<int> *d_hist;
  MLCommon::host_buffer<int> *h_hist;

  //Host/Device histograms and device minmaxs
  MLCommon::device_buffer<T> *d_globalminmax;
  MLCommon::device_buffer<int> *d_histout;
  MLCommon::device_buffer<unsigned int> *d_colids;
  MLCommon::host_buffer<int> *h_histout;
  MLCommon::device_buffer<T> *d_mseout, *d_predout;
  MLCommon::host_buffer<T> *h_mseout, *h_predout;

  //Below pointers are shared for split functions
  MLCommon::device_buffer<char> *d_flags_left, *d_flags_right;
  MLCommon::host_buffer<int> *nrowsleftright;
  MLCommon::device_buffer<char> *d_split_temp_storage = nullptr;
  size_t split_temp_storage_bytes = 0;

  MLCommon::device_buffer<int> *d_num_selected_out;
  MLCommon::device_buffer<unsigned int> *temprowids;
  MLCommon::device_buffer<T> *question_value, *temp_data;

  //Total temp mem
  size_t totalmem = 0;

  //CUDA stream
  cudaStream_t stream;

  //For quantiles
  MLCommon::device_buffer<T> *d_quantile = nullptr;

  const ML::cumlHandle_impl &ml_handle;

  TemporaryMemory(const ML::cumlHandle_impl &handle, int N, int Ncols,
                  int maxstr, int n_unique, int n_bins, const int split_algo);

  void print_info();
  ~TemporaryMemory();
};
