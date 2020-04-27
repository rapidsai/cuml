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
#include <cuml/common/cuml_allocator.hpp>
#include "common/cumlHandle.hpp"

template <class T, class L>
struct TemporaryMemory {
  //depth algorithm changer
  const int swap_depth = 14;
  static const int gather_threads = 256;
  size_t parentsz, childsz, gather_max_nodes;
  //Allocators parsed from CUML handle
  std::shared_ptr<MLCommon::deviceAllocator> device_allocator;
  std::shared_ptr<MLCommon::hostAllocator> host_allocator;

  //Tree holder for gather algorithm
  MLCommon::device_buffer<SparseTreeNode<T, L>> *d_sparsenodes = nullptr;
  MLCommon::host_buffer<SparseTreeNode<T, L>> *h_sparsenodes = nullptr;

  //Temporary data buffer
  MLCommon::device_buffer<T> *temp_data = nullptr;
  //Temporary CUB buffer
  MLCommon::device_buffer<char> *temp_cub_buffer = nullptr;
  size_t temp_cub_bytes;
  MLCommon::host_buffer<int> *h_counter = nullptr;
  MLCommon::device_buffer<int> *d_counter = nullptr;
  //Host/Device histograms and device minmaxs
  MLCommon::device_buffer<T> *d_globalminmax = nullptr;
  MLCommon::host_buffer<T> *h_globalminmax = nullptr;
  MLCommon::device_buffer<T> *d_mseout = nullptr;
  MLCommon::device_buffer<T> *d_predout = nullptr;
  MLCommon::host_buffer<T> *h_mseout = nullptr;
  MLCommon::host_buffer<T> *h_predout = nullptr;
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
  MLCommon::device_buffer<unsigned int> *d_colstart = nullptr;
  MLCommon::host_buffer<unsigned int> *h_colids = nullptr;
  MLCommon::host_buffer<unsigned int> *h_colstart = nullptr;
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
  TemporaryMemory(
    const std::shared_ptr<MLCommon::deviceAllocator> device_allocator_in,
    const std::shared_ptr<MLCommon::hostAllocator> host_allocator_in,
    const cudaStream_t stream_in, int N, int Ncols, float colper, int n_unique,
    int n_bins, const int split_algo, int depth, bool col_shuffle);
  TemporaryMemory(const ML::cumlHandle_impl &handle, cudaStream_t stream_in,
                  int N, int Ncols, float colper, int n_unique, int n_bins,
                  const int split_algo, int depth, bool colshuffle);
  ~TemporaryMemory();
  void LevelMemAllocator(int nrows, int ncols, float colper, int n_unique,
                         int nbins, int depth, const int split_algo,
                         bool col_shuffle);

  void LevelMemCleaner();
  void print_info(int depth, int nrows, int ncols, float colper);
};
#include "memory.cuh"
