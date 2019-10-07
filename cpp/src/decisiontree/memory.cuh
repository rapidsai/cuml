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
TemporaryMemory<T, L>::TemporaryMemory(
  const std::shared_ptr<MLCommon::deviceAllocator> device_allocator_in,
  const std::shared_ptr<MLCommon::hostAllocator> host_allocator_in,
  const cudaStream_t stream_in, int N, int Ncols, float colper, int n_unique,
  int n_bins, const int split_algo, int depth, bool col_shuffle) {
  stream = stream_in;
  splitalgo = split_algo;

  max_shared_mem = MLCommon::getSharedMemPerBlock();
  num_sms = MLCommon::getMultiProcessorCount();
  device_allocator = device_allocator_in;
  host_allocator = host_allocator_in;
  LevelMemAllocator(N, Ncols, colper, n_unique, n_bins, depth, split_algo,
                    col_shuffle);
}

template <class T, class L>
TemporaryMemory<T, L>::TemporaryMemory(const ML::cumlHandle_impl& handle,
                                       cudaStream_t stream_in, int N, int Ncols,
                                       float colper, int n_unique, int n_bins,
                                       const int split_algo, int depth,
                                       bool col_shuffle) {
  //Assign Stream from cumlHandle
  stream = stream_in;
  splitalgo = split_algo;

  max_shared_mem = MLCommon::getSharedMemPerBlock();
  num_sms = MLCommon::getMultiProcessorCount();
  device_allocator = handle.getDeviceAllocator();
  host_allocator = handle.getHostAllocator();
  LevelMemAllocator(N, Ncols, colper, n_unique, n_bins, depth, split_algo,
                    col_shuffle);
}

template <class T, class L>
TemporaryMemory<T, L>::~TemporaryMemory() {
  LevelMemCleaner();
}

template <class T, class L>
void TemporaryMemory<T, L>::print_info() {
  std::cout << " Total temporary memory usage--> "
            << ((double)totalmem / (1024 * 1024)) << "  MB" << std::endl;
}

template <class T, class L>
void TemporaryMemory<T, L>::LevelMemAllocator(int nrows, int ncols,
                                              float colper, int n_unique,
                                              int nbins, int depth,
                                              const int split_algo,
                                              bool col_shuffle) {
  if (depth > 20 || (depth == -1)) {
    max_nodes_per_level = pow(2, 20);
  } else {
    max_nodes_per_level = pow(2, depth);
  }
  int maxnodes = max_nodes_per_level;
  int ncols_sampled = (int)(ncols * colper);
  d_flags =
    new MLCommon::device_buffer<unsigned int>(device_allocator, stream, nrows);
  h_split_colidx =
    new MLCommon::host_buffer<int>(host_allocator, stream, maxnodes);
  h_split_binidx =
    new MLCommon::host_buffer<int>(host_allocator, stream, maxnodes);
  d_split_colidx =
    new MLCommon::device_buffer<int>(device_allocator, stream, maxnodes);
  d_split_binidx =
    new MLCommon::device_buffer<int>(device_allocator, stream, maxnodes);
  h_new_node_flags =
    new MLCommon::host_buffer<unsigned int>(host_allocator, stream, maxnodes);
  d_new_node_flags = new MLCommon::device_buffer<unsigned int>(
    device_allocator, stream, maxnodes);
  h_parent_metric =
    new MLCommon::host_buffer<T>(host_allocator, stream, maxnodes);
  h_child_best_metric =
    new MLCommon::host_buffer<T>(host_allocator, stream, 2 * maxnodes);
  h_outgain =
    new MLCommon::host_buffer<float>(host_allocator, stream, maxnodes);
  d_parent_metric =
    new MLCommon::device_buffer<T>(device_allocator, stream, maxnodes);
  d_child_best_metric =
    new MLCommon::device_buffer<T>(device_allocator, stream, 2 * maxnodes);
  d_outgain =
    new MLCommon::device_buffer<float>(device_allocator, stream, maxnodes);
  if (split_algo == 0) {
    d_globalminmax = new MLCommon::device_buffer<T>(
      device_allocator, stream, 2 * maxnodes * ncols_sampled);
    h_globalminmax = new MLCommon::host_buffer<T>(host_allocator, stream,
                                                  2 * maxnodes * ncols_sampled);
    totalmem += maxnodes * ncols * sizeof(T);
  } else {
    h_quantile =
      new MLCommon::host_buffer<T>(host_allocator, stream, nbins * ncols);
    d_quantile =
      new MLCommon::device_buffer<T>(device_allocator, stream, nbins * ncols);
    totalmem += nbins * ncols * sizeof(T);
  }
  d_sample_cnt =
    new MLCommon::device_buffer<unsigned int>(device_allocator, stream, nrows);
  if (col_shuffle == true) {
    d_colids = new MLCommon::device_buffer<unsigned int>(
      device_allocator, stream, ncols_sampled * maxnodes);
    h_colids = new MLCommon::host_buffer<unsigned int>(
      host_allocator, stream, ncols_sampled * maxnodes);

  } else {
    d_colids = new MLCommon::device_buffer<unsigned int>(device_allocator,
                                                         stream, ncols);
    d_colstart = new MLCommon::device_buffer<unsigned int>(device_allocator,
                                                           stream, maxnodes);
    h_colids =
      new MLCommon::host_buffer<unsigned int>(host_allocator, stream, ncols);
    h_colstart =
      new MLCommon::host_buffer<unsigned int>(host_allocator, stream, maxnodes);
  }
  totalmem += nrows * 2 * sizeof(unsigned int);
  totalmem += maxnodes * 3 * sizeof(int);
  totalmem += maxnodes * sizeof(float);
  totalmem += 3 * maxnodes * sizeof(T);
  totalmem += (ncols + maxnodes) * sizeof(int);
  //Regression
  if (typeid(L) == typeid(T)) {
    d_mseout = new MLCommon::device_buffer<T>(
      device_allocator, stream, 2 * nbins * ncols_sampled * maxnodes);
    d_predout = new MLCommon::device_buffer<T>(
      device_allocator, stream, nbins * ncols_sampled * maxnodes);
    d_count = new MLCommon::device_buffer<unsigned int>(
      device_allocator, stream, nbins * ncols_sampled * maxnodes);
    d_parent_pred =
      new MLCommon::device_buffer<T>(device_allocator, stream, maxnodes);
    d_parent_count = new MLCommon::device_buffer<unsigned int>(
      device_allocator, stream, maxnodes);
    d_child_pred =
      new MLCommon::device_buffer<T>(device_allocator, stream, 2 * maxnodes);
    d_child_count = new MLCommon::device_buffer<unsigned int>(
      device_allocator, stream, 2 * maxnodes);
    h_mseout = new MLCommon::host_buffer<T>(
      host_allocator, stream, 2 * nbins * ncols_sampled * maxnodes);
    h_predout = new MLCommon::host_buffer<T>(host_allocator, stream,
                                             nbins * ncols_sampled * maxnodes);
    h_count = new MLCommon::host_buffer<unsigned int>(
      host_allocator, stream, nbins * ncols_sampled * maxnodes);
    h_child_pred =
      new MLCommon::host_buffer<T>(host_allocator, stream, 2 * maxnodes);
    h_child_count = new MLCommon::host_buffer<unsigned int>(
      host_allocator, stream, 2 * maxnodes);

    totalmem += 3 * nbins * ncols_sampled * maxnodes * sizeof(T);
    totalmem += nbins * ncols_sampled * maxnodes * sizeof(unsigned int);
    totalmem += 3 * maxnodes * sizeof(T);
    totalmem += 3 * maxnodes * sizeof(unsigned int);
  }

  //Classification
  if (typeid(L) == typeid(int)) {
    size_t histcount = ncols_sampled * nbins * n_unique * maxnodes;
    d_histogram = new MLCommon::device_buffer<unsigned int>(device_allocator,
                                                            stream, histcount);
    h_histogram = new MLCommon::host_buffer<unsigned int>(host_allocator,
                                                          stream, histcount);
    h_parent_hist = new MLCommon::host_buffer<unsigned int>(
      host_allocator, stream, maxnodes * n_unique);
    h_child_hist = new MLCommon::host_buffer<unsigned int>(
      host_allocator, stream, 2 * maxnodes * n_unique);
    d_parent_hist = new MLCommon::device_buffer<unsigned int>(
      device_allocator, stream, maxnodes * n_unique);
    d_child_hist = new MLCommon::device_buffer<unsigned int>(
      device_allocator, stream, 2 * maxnodes * n_unique);
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
  if (split_algo == ML::SPLIT_ALGO::HIST) {
    size_t shmem_per_node = 2 * sizeof(T);
    max_nodes_minmax = max_shared_mem / shmem_per_node;
    max_nodes_minmax /= 2;
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
  if (h_quantile != nullptr) h_quantile->release(stream);
  if (d_quantile != nullptr) d_quantile->release(stream);
  if (d_globalminmax != nullptr) d_globalminmax->release(stream);
  if (h_globalminmax != nullptr) h_globalminmax->release(stream);
  d_sample_cnt->release(stream);
  d_colids->release(stream);
  if (d_colstart != nullptr) d_colstart->release(stream);
  h_colids->release(stream);
  if (h_colstart != nullptr) h_colstart->release(stream);
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
  if (h_quantile != nullptr) delete h_quantile;
  if (d_quantile != nullptr) delete d_quantile;
  if (d_globalminmax != nullptr) delete d_globalminmax;
  if (h_globalminmax != nullptr) delete h_globalminmax;
  delete d_sample_cnt;
  delete d_colids;
  delete h_colids;
  if (d_colstart != nullptr) delete d_colstart;
  if (h_colstart != nullptr) delete h_colstart;
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
    d_child_pred->release(stream);
    d_child_count->release(stream);

    h_child_pred->release(stream);
    h_child_count->release(stream);
    h_mseout->release(stream);
    h_predout->release(stream);
    h_count->release(stream);

    delete d_child_pred;
    delete d_child_count;
    delete d_parent_pred;
    delete d_parent_count;
    delete d_mseout;
    delete d_predout;
    delete d_count;

    delete h_mseout;
    delete h_predout;
    delete h_count;
    delete h_child_pred;
    delete h_child_count;
  }
}
