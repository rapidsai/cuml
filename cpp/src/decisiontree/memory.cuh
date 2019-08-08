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
                                       int Ncols, int n_unique, int n_bins,
                                       const int split_algo, int depth)
  : ml_handle(handle) {
  //Assign Stream from cumlHandle
  stream = ml_handle.getStream();
  splitalgo = split_algo;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, ml_handle.getDevice()));
  max_shared_mem = prop.sharedMemPerBlock;
  num_sms = prop.multiProcessorCount;

  if (splitalgo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) {
    LevelMemAllocator(N, Ncols, n_unique, n_bins, depth);
  } else {
    NodeMemAllocator(N, Ncols, n_unique, n_bins, split_algo);
  }
}

template <class T, class L>
TemporaryMemory<T, L>::~TemporaryMemory() {
  if (splitalgo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) {
    LevelMemCleaner();
  } else {
    NodeMemCleaner();
  }
}

template <class T, class L>
void TemporaryMemory<T, L>::NodeMemAllocator(int N, int Ncols, int n_unique,
                                             int n_bins, const int split_algo) {
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
}

template <class T, class L>
void TemporaryMemory<T, L>::print_info() {
  std::cout << " Total temporary memory usage--> "
            << ((double)totalmem / (1024 * 1024)) << "  MB" << std::endl;
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

template <class T, class L>
void TemporaryMemory<T, L>::LevelMemAllocator(int nrows, int ncols,
                                              int n_unique, int nbins,
                                              int depth) {
  if (depth > 20) {
    max_nodes_per_level = pow(2, 20);
  } else {
    max_nodes_per_level = pow(2, depth);
  }
  int maxnodes = max_nodes_per_level;

  d_flags = new MLCommon::device_buffer<unsigned int>(
    ml_handle.getDeviceAllocator(), stream, nrows);
  h_split_colidx = new MLCommon::host_buffer<int>(ml_handle.getHostAllocator(),
                                                  stream, maxnodes);
  h_split_binidx = new MLCommon::host_buffer<int>(ml_handle.getHostAllocator(),
                                                  stream, maxnodes);
  d_split_colidx = new MLCommon::device_buffer<int>(
    ml_handle.getDeviceAllocator(), stream, maxnodes);
  d_split_binidx = new MLCommon::device_buffer<int>(
    ml_handle.getDeviceAllocator(), stream, maxnodes);
  h_new_node_flags = new MLCommon::host_buffer<unsigned int>(
    ml_handle.getHostAllocator(), stream, maxnodes);
  d_new_node_flags = new MLCommon::device_buffer<unsigned int>(
    ml_handle.getDeviceAllocator(), stream, maxnodes);
  h_parent_metric = new MLCommon::host_buffer<T>(ml_handle.getHostAllocator(),
                                                 stream, maxnodes);
  h_child_best_metric = new MLCommon::host_buffer<T>(
    ml_handle.getHostAllocator(), stream, 2 * maxnodes);
  h_outgain = new MLCommon::host_buffer<float>(ml_handle.getHostAllocator(),
                                               stream, maxnodes);
  d_parent_metric = new MLCommon::device_buffer<T>(
    ml_handle.getDeviceAllocator(), stream, maxnodes);
  d_child_best_metric = new MLCommon::device_buffer<T>(
    ml_handle.getDeviceAllocator(), stream, 2 * maxnodes);
  d_outgain = new MLCommon::device_buffer<float>(ml_handle.getDeviceAllocator(),
                                                 stream, maxnodes);
  h_quantile = new MLCommon::host_buffer<T>(ml_handle.getHostAllocator(),
                                            stream, nbins * ncols);
  d_quantile = new MLCommon::device_buffer<T>(ml_handle.getDeviceAllocator(),
                                              stream, nbins * ncols);
  d_sample_cnt = new MLCommon::device_buffer<unsigned int>(
    ml_handle.getDeviceAllocator(), stream, nrows);
  d_colids = new MLCommon::device_buffer<unsigned int>(
    ml_handle.getDeviceAllocator(), stream, ncols);

  totalmem += nrows * 2 * sizeof(unsigned int);
  totalmem += maxnodes * 3 * sizeof(int);
  totalmem += maxnodes * sizeof(float);
  totalmem += 3 * maxnodes * sizeof(T);
  totalmem += ncols * sizeof(int);
  totalmem += nbins * ncols * sizeof(T);
  //Regression
  if (typeid(L) == typeid(T)) {
    d_mseout = new MLCommon::device_buffer<T>(
      ml_handle.getDeviceAllocator(), stream, 2 * nbins * ncols * maxnodes);
    d_predout = new MLCommon::device_buffer<T>(
      ml_handle.getDeviceAllocator(), stream, nbins * ncols * maxnodes);
    d_count = new MLCommon::device_buffer<unsigned int>(
      ml_handle.getDeviceAllocator(), stream, nbins * ncols * maxnodes);
    d_parent_pred = new MLCommon::device_buffer<T>(
      ml_handle.getDeviceAllocator(), stream, maxnodes);
    d_parent_count = new MLCommon::device_buffer<unsigned int>(
      ml_handle.getDeviceAllocator(), stream, maxnodes);
    d_child_pred = new MLCommon::device_buffer<T>(
      ml_handle.getDeviceAllocator(), stream, 2 * maxnodes);
    d_child_count = new MLCommon::device_buffer<unsigned int>(
      ml_handle.getDeviceAllocator(), stream, 2 * maxnodes);
    h_mseout = new MLCommon::host_buffer<T>(
      ml_handle.getHostAllocator(), stream, 2 * nbins * ncols * maxnodes);
    h_predout = new MLCommon::host_buffer<T>(ml_handle.getHostAllocator(),
                                             stream, nbins * ncols * maxnodes);
    h_count = new MLCommon::host_buffer<unsigned int>(
      ml_handle.getHostAllocator(), stream, nbins * ncols * maxnodes);
    h_child_pred = new MLCommon::host_buffer<T>(ml_handle.getHostAllocator(),
                                                stream, 2 * maxnodes);
    h_child_count = new MLCommon::host_buffer<unsigned int>(
      ml_handle.getHostAllocator(), stream, 2 * maxnodes);

    totalmem += 3 * nbins * ncols * maxnodes * sizeof(T);
    totalmem += nbins * ncols * maxnodes * sizeof(unsigned int);
    totalmem += 3 * maxnodes * sizeof(T);
    totalmem += 3 * maxnodes * sizeof(unsigned int);
  }

  //Classification
  if (typeid(L) == typeid(int)) {
    size_t histcount = ncols * nbins * n_unique * maxnodes;
    d_histogram = new MLCommon::device_buffer<unsigned int>(
      ml_handle.getDeviceAllocator(), stream, histcount);
    h_histogram = new MLCommon::host_buffer<unsigned int>(
      ml_handle.getHostAllocator(), stream, histcount);
    h_parent_hist = new MLCommon::host_buffer<unsigned int>(
      ml_handle.getHostAllocator(), stream, maxnodes * n_unique);
    h_child_hist = new MLCommon::host_buffer<unsigned int>(
      ml_handle.getHostAllocator(), stream, 2 * maxnodes * n_unique);
    d_parent_hist = new MLCommon::device_buffer<unsigned int>(
      ml_handle.getDeviceAllocator(), stream, maxnodes * n_unique);
    d_child_hist = new MLCommon::device_buffer<unsigned int>(
      ml_handle.getDeviceAllocator(), stream, 2 * maxnodes * n_unique);
    totalmem += histcount * sizeof(unsigned int);
    totalmem += n_unique * maxnodes * 3 * sizeof(unsigned int);
  }
  //Calculate Max nodes in shared memory.
  if (typeid(L) == typeid(int)) {
    max_nodes_class = max_shared_mem / (nbins * n_unique * sizeof(int));
    max_nodes_class /= 2;  // For occupancy purposes.
  }
  if (typeid(L) == typeid(T)) {
    size_t pernode_pred = nbins * (sizeof(T) + sizeof(unsigned int));
    max_nodes_pred = max_shared_mem / pernode_pred;
    max_nodes_mse = max_shared_mem / (pernode_pred + 2 * nbins * sizeof(T));
    max_nodes_pred /= 2;  // For occupancy purposes.
    max_nodes_mse /= 2;   // For occupancy purposes.
  }
}

template <class T, class L>
void TemporaryMemory<T, L>::LevelMemCleaner() {
  h_new_node_flags->release(stream);
  d_new_node_flags->release(stream);
  h_split_colidx->release(stream);
  d_split_colidx->release(stream);
  h_split_binidx->release(stream);
  d_split_binidx->release(stream);
  h_parent_metric->release(stream);
  h_child_best_metric->release(stream);
  h_outgain->release(stream);
  d_parent_metric->release(stream);
  d_child_best_metric->release(stream);
  d_outgain->release(stream);
  d_flags->release(stream);
  h_quantile->release(stream);
  d_quantile->release(stream);
  d_sample_cnt->release(stream);
  d_colids->release(stream);
  delete h_new_node_flags;
  delete d_new_node_flags;
  delete h_split_colidx;
  delete d_split_colidx;
  delete h_split_binidx;
  delete d_split_binidx;
  delete h_parent_metric;
  delete h_child_best_metric;
  delete h_outgain;
  delete d_parent_metric;
  delete d_child_best_metric;
  delete d_outgain;
  delete d_flags;
  delete h_quantile;
  delete d_quantile;
  delete d_sample_cnt;
  delete d_colids;
  //Classification
  if (typeid(L) == typeid(int)) {
    h_histogram->release(stream);
    d_histogram->release(stream);
    h_parent_hist->release(stream);
    h_child_hist->release(stream);
    d_parent_hist->release(stream);
    d_child_hist->release(stream);
    delete d_histogram;
    delete h_histogram;
    delete h_parent_hist;
    delete h_child_hist;
    delete d_parent_hist;
    delete d_child_hist;
  }
  //Regression
  if (typeid(L) == typeid(T)) {
    d_parent_pred->release(stream);
    d_parent_count->release(stream);
    d_mseout->release(stream);
    d_predout->release(stream);
    d_count->release(stream);
    h_mseout->release(stream);
    h_predout->release(stream);
    h_count->release(stream);
    delete d_parent_pred;
    delete d_parent_count;
    delete d_mseout;
    delete d_predout;
    delete d_count;
    delete h_mseout;
    delete h_predout;
    delete h_count;
  }
}
