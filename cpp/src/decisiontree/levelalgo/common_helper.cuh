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
#include <cuml/tree/flatnode.h>
#include "common_kernel.cuh"
#include "random/rng.h"
#include "stats/minmax.h"

//GPU based RNG in the range [0,Ncols)
void random_startids(unsigned int *data, int len, int Ncols, const int num_sms,
                     cudaStream_t stream) {
  uint64_t offset = 0;
  MLCommon::Random::randImpl(
    offset, data, len,
    [=] __device__(unsigned int val, int idx) { return (val % Ncols); }, 256,
    4 * num_sms, MLCommon::Random::GeneratorType::GenKiss99, stream);
}

/*This functions does feature subsampling.
 *The default is reshuffling of a feature list at ever level followed by random start index in the reshuffled vector for each node.
 *In case full reshuffle is enabled. A reshuffle is performed for every node in the tree
 */
template <typename T, typename L, typename RNG, typename DIST>
void update_feature_sampling(unsigned int *h_colids, unsigned int *d_colids,
                             unsigned int *h_colstart, unsigned int *d_colstart,
                             const int Ncols, const int ncols_sampled,
                             const int n_nodes, RNG rng, DIST dist,
                             std::vector<unsigned int> &feature_selector,
                             std::shared_ptr<TemporaryMemory<T, L>> tempmem) {
  if (h_colstart != nullptr) {
    if (Ncols != ncols_sampled) {
      std::shuffle(h_colids, h_colids + Ncols, rng);
      MLCommon::updateDevice(d_colids, h_colids, Ncols, tempmem->stream);
      if (n_nodes < 256 * tempmem->num_sms) {
        for (int i = 0; i < n_nodes; i++) {
          h_colstart[i] = dist(rng);
        }
        MLCommon::updateDevice(d_colstart, h_colstart, n_nodes,
                               tempmem->stream);
      } else {
        random_startids(d_colstart, n_nodes, Ncols, tempmem->num_sms,
                        tempmem->stream);
        MLCommon::updateHost(h_colstart, d_colstart, n_nodes, tempmem->stream);
      }
    }
  } else {
    for (int i = 0; i < n_nodes; i++) {
      std::vector<unsigned int> temp(feature_selector);
      std::shuffle(temp.begin(), temp.end(), rng);
      memcpy(&h_colids[i * ncols_sampled], temp.data(),
             ncols_sampled * sizeof(unsigned int));
    }
    MLCommon::updateDevice(d_colids, h_colids, ncols_sampled * n_nodes,
                           tempmem->stream);
  }
}

//This function calcualtes min/max from the samples that belong in a given node. This is done for all the nodes at a given level
template <typename T>
void get_minmax(const T *data, const unsigned int *flags,
                const unsigned int *colids, const unsigned int *colstart,
                const int nrows, const int Ncols, const int ncols_sampled,
                const int n_nodes, const int max_shmem_nodes, T *d_minmax,
                T *h_minmax, cudaStream_t &stream) {
  using E = typename MLCommon::Stats::encode_traits<T>::E;
  T init_val = std::numeric_limits<T>::max();
  int threads = 128;
  int nblocks = MLCommon::ceildiv(2 * ncols_sampled * n_nodes, threads);
  minmax_init_kernel<T, E><<<nblocks, threads, 0, stream>>>(
    d_minmax, ncols_sampled * n_nodes, n_nodes, init_val);
  CUDA_CHECK(cudaGetLastError());

  nblocks = MLCommon::ceildiv(nrows, threads);
  if (n_nodes <= max_shmem_nodes) {
    get_minmax_kernel<T, E>
      <<<nblocks, threads, 2 * n_nodes * sizeof(T), stream>>>(
        data, flags, colids, colstart, nrows, Ncols, ncols_sampled, n_nodes,
        init_val, d_minmax);
  } else {
    get_minmax_kernel_global<T, E><<<nblocks, threads, 0, stream>>>(
      data, flags, colids, colstart, nrows, Ncols, ncols_sampled, n_nodes,
      d_minmax);
  }
  CUDA_CHECK(cudaGetLastError());

  nblocks = MLCommon::ceildiv(2 * ncols_sampled * n_nodes, threads);
  minmax_decode_kernel<T, E>
    <<<nblocks, threads, 0, stream>>>(d_minmax, ncols_sampled * n_nodes);

  CUDA_CHECK(cudaGetLastError());
  MLCommon::updateHost(h_minmax, d_minmax, 2 * n_nodes * ncols_sampled, stream);
}
// This function does setup for flags. and count.
void setup_sampling(unsigned int *flagsptr, unsigned int *sample_cnt,
                    const unsigned int *rowids, const int nrows,
                    const int n_sampled_rows, cudaStream_t &stream) {
  CUDA_CHECK(cudaMemsetAsync(sample_cnt, 0, nrows * sizeof(int), stream));
  int threads = 256;
  int blocks = MLCommon::ceildiv(n_sampled_rows, threads);
  setup_counts_kernel<<<blocks, threads, 0, stream>>>(sample_cnt, rowids,
                                                      n_sampled_rows);
  CUDA_CHECK(cudaGetLastError());
  blocks = MLCommon::ceildiv(nrows, threads);
  setup_flags_kernel<<<blocks, threads, 0, stream>>>(sample_cnt, flagsptr,
                                                     nrows);
  CUDA_CHECK(cudaGetLastError());
}

//This function call the split kernel
template <typename T, typename L>
void make_level_split(const T *data, const int nrows, const int Ncols,
                      const int ncols_sampled, const int nbins,
                      const int n_nodes, const int split_algo,
                      int *split_colidx, int *split_binidx,
                      const unsigned int *new_node_flags, unsigned int *flags,
                      std::shared_ptr<TemporaryMemory<T, L>> tempmem) {
  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);
  unsigned int *d_colstart = nullptr;
  if (tempmem->d_colstart != nullptr) d_colstart = tempmem->d_colstart->data();
  if (split_algo == 0) {
    split_level_kernel<T, MinMaxQues<T>>
      <<<blocks, threads, 0, tempmem->stream>>>(
        data, tempmem->d_globalminmax->data(), tempmem->d_colids->data(),
        d_colstart, split_colidx, split_binidx, nrows, Ncols, ncols_sampled,
        nbins, n_nodes, new_node_flags, flags);
  } else {
    split_level_kernel<T, QuantileQues<T>>
      <<<blocks, threads, 0, tempmem->stream>>>(
        data, tempmem->d_quantile->data(), tempmem->d_colids->data(),
        d_colstart, split_colidx, split_binidx, nrows, Ncols, ncols_sampled,
        nbins, n_nodes, new_node_flags, flags);
  }
  CUDA_CHECK(cudaGetLastError());
}

/* node_hist[i] holds the # times label i appear in current data. The vector is computed during gini
   computation. */
int get_class_hist(unsigned int *node_hist, const int n_unique_labels) {
  unsigned int maxval = node_hist[0];
  int classval = 0;
  for (int i = 1; i < n_unique_labels; i++) {
    if (node_hist[i] > maxval) {
      maxval = node_hist[i];
      classval = i;
    }
  }
  return classval;
}

template <typename T>
T getQuesValue(const T *minmax, const T *quantile, const int nbins,
               const int colid, const int binid, const int nodeid,
               const int n_nodes, const int featureid, const int split_algo) {
  if (split_algo == 0) {
    T min = minmax[nodeid + colid * n_nodes * 2];
    T delta = (minmax[nodeid + n_nodes + colid * n_nodes * 2] - min) / nbins;
    return (min + delta * (binid + 1));
  } else {
    return quantile[featureid * nbins + binid];
  }
}

unsigned int getQuesColumn(const unsigned int *colids, const int colstart_local,
                           const int Ncols, const int ncols_sampled,
                           const int colidx, const int nodeid) {
  unsigned int col;
  if (colstart_local != -1) {
    col = colids[(colstart_local + colidx) % Ncols];
  } else {
    col = colids[nodeid * ncols_sampled + colidx];
  }
  return col;
}
