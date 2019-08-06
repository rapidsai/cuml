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

  //No of SMs
  int num_sms;

  //Maximum shared memory in GPU
  size_t max_shared_mem;

  //For quantiles and colids; this part is common
  MLCommon::device_buffer<T> *d_quantile = nullptr;
  MLCommon::host_buffer<T> *h_quantile = nullptr;
  MLCommon::device_buffer<unsigned int> *d_colids = nullptr;

  const ML::cumlHandle_impl &ml_handle;
  //Split algo
  int splitalgo;

  //For level algorithm
  MLCommon::device_buffer<unsigned int> *d_flags = nullptr;
  MLCommon::device_buffer<unsigned int> *d_histogram = nullptr;
  MLCommon::host_buffer<unsigned int> *h_histogram = nullptr;
  MLCommon::host_buffer<int> *h_split_colidx = nullptr;
  MLCommon::host_buffer<int> *h_split_binidx = nullptr;
  MLCommon::device_buffer<int> *d_split_colidx = nullptr;
  MLCommon::device_buffer<int> *d_split_binidx = nullptr;
  MLCommon::host_buffer<unsigned int> *h_new_node_flags = nullptr;
  MLCommon::device_buffer<unsigned int> *d_new_node_flags = nullptr;
  MLCommon::host_buffer<unsigned int> *h_parent_hist = nullptr;
  MLCommon::host_buffer<unsigned int> *h_child_hist = nullptr;
  MLCommon::device_buffer<unsigned int> *d_parent_hist = nullptr;
  MLCommon::device_buffer<unsigned int> *d_child_hist = nullptr;
  MLCommon::host_buffer<T> *h_parent_metric = nullptr;
  MLCommon::host_buffer<T> *h_child_best_metric = nullptr;
  MLCommon::host_buffer<float> *h_outgain = nullptr;
  MLCommon::device_buffer<float> *d_outgain = nullptr;
  MLCommon::device_buffer<T> *d_parent_metric = nullptr;
  MLCommon::device_buffer<T> *d_child_best_metric = nullptr;
  MLCommon::device_buffer<unsigned int> *d_sample_cnt = nullptr;

  MLCommon::device_buffer<T> *d_parent_pred = nullptr;
  MLCommon::device_buffer<unsigned int> *d_parent_count = nullptr;
  MLCommon::device_buffer<T> *d_child_pred = nullptr;
  MLCommon::device_buffer<unsigned int> *d_child_count = nullptr;
  MLCommon::device_buffer<unsigned int> *d_count = nullptr;
  MLCommon::host_buffer<unsigned int> *h_count = nullptr;
  MLCommon::host_buffer<T> *h_child_pred = nullptr;
  MLCommon::host_buffer<unsigned int> *h_child_count = nullptr;

  int max_nodes_class = 0;
  int max_nodes_pred = 0;
  int max_nodes_mse = 0;
  int max_nodes_per_level = 0;
  int max_nodes_minmax = 0;
  TemporaryMemory(const ML::cumlHandle_impl &handle, int N, int Ncols,
                  int n_unique, int n_bins, const int split_algo, int depth);
  ~TemporaryMemory();
  void NodeMemAllocator(int N, int Ncols, int n_unique, int n_bins,
                        const int split_algo);
  void LevelMemAllocator(int nrows, int ncols, int n_unique, int nbins,
                         int depth, const int split_algo);
  void NodeMemCleaner();
  void LevelMemCleaner();
  void print_info();
};
#include "memory.cuh"
