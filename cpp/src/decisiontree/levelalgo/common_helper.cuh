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
#include "common_kernel.cuh"
#include "flatnode.h"
#include "stats/minmax.h"

template <typename T>
void get_minmax(const T *data, const unsigned int *flags,
                const unsigned int *colids, const int nrows, const int ncols,
                const int n_nodes, const int max_shmem_nodes, T *d_minmax,
                T *h_minmax, cudaStream_t &stream) {
  using E = typename MLCommon::Stats::encode_traits<T>::E;
  T init_val = std::numeric_limits<T>::max();
  init_val = 100;
  int threads = 128;
  int nblocks = MLCommon::ceildiv(2 * ncols * n_nodes, threads);
  minmax_init_kernel<T, E><<<nblocks, threads, 0, stream>>>(
    d_minmax, ncols * n_nodes, n_nodes, init_val);
  CUDA_CHECK(cudaGetLastError());

  nblocks = MLCommon::ceildiv(nrows, threads);
  if (n_nodes <= max_shmem_nodes) {
    get_minmax_kernel<T, E>
      <<<nblocks, threads, 2 * n_nodes * sizeof(T), stream>>>(
        data, flags, colids, nrows, ncols, n_nodes, init_val, d_minmax);
  } else {
    get_minmax_kernel_global<T, E><<<nblocks, threads, 0, stream>>>(
      data, flags, colids, nrows, 1, n_nodes, d_minmax);
  }
  CUDA_CHECK(cudaGetLastError());

  nblocks = MLCommon::ceildiv(2 * ncols * n_nodes, threads);
  minmax_decode_kernel<T, E>
    <<<nblocks, threads, 0, stream>>>(d_minmax, ncols * n_nodes);

  CUDA_CHECK(cudaGetLastError());
  MLCommon::updateHost(h_minmax, d_minmax, 2 * n_nodes * ncols, stream);
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
void make_level_split(const T *data, const int nrows, const int ncols,
                      const int nbins, const int n_nodes, const int split_algo,
                      int *split_colidx, int *split_binidx,
                      const unsigned int *new_node_flags, unsigned int *flags,
                      std::shared_ptr<TemporaryMemory<T, L>> tempmem) {
  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);
  if (split_algo == 0) {
    split_level_kernel<T, MinMaxQues<T>>
      <<<blocks, threads, 0, tempmem->stream>>>(
        data, tempmem->d_globalminmax->data(), tempmem->d_colids->data(),
        split_colidx, split_binidx, nrows, ncols, nbins, n_nodes,
        new_node_flags, flags);
  } else {
    split_level_kernel<T, QuantileQues<T>>
      <<<blocks, threads, 0, tempmem->stream>>>(
        data, tempmem->d_quantile->data(), tempmem->d_colids->data(),
        split_colidx, split_binidx, nrows, ncols, nbins, n_nodes,
        new_node_flags, flags);
  }
  CUDA_CHECK(cudaGetLastError());
}

// Converts flat sparse tree generated to recursive format.
template <typename T, typename L>
ML::DecisionTree::TreeNode<T, L> *go_recursive_sparse(
  std::vector<SparseTreeNode<T, L>> &sparsetree, int idx = 0) {
  ML::DecisionTree::TreeNode<T, L> *node = NULL;
  node = new ML::DecisionTree::TreeNode<T, L>();
  node->split_metric_val = sparsetree[idx].best_metric_val;
  node->question.column = sparsetree[idx].colid;
  node->question.value = sparsetree[idx].quesval;
  node->prediction = sparsetree[idx].prediction;
  if (sparsetree[idx].colid == -1) {
    return node;
  }
  node->left = go_recursive_sparse(sparsetree, sparsetree[idx].left_child_id);
  node->right =
    go_recursive_sparse(sparsetree, sparsetree[idx].left_child_id + 1);
  return node;
}

/* node_hist[i] holds the # times label i appear in current data. The vector is computed during gini
   computation. */
int get_class_hist(std::vector<int> &node_hist) {
  int classval =
    std::max_element(node_hist.begin(), node_hist.end()) - node_hist.begin();
  return classval;
}

template <typename T>
T getQuesValue(const T *minmax, const T *quantile, const int nbins,
               const int colid, const int binid, const int nodeid,
               const int n_nodes, const std::vector<unsigned int> &colselector,
               const int split_algo) {
  if (split_algo == 0) {
    T min = minmax[nodeid + colid * n_nodes * 2];
    T delta = (minmax[nodeid + n_nodes + colid * n_nodes * 2] - min) / nbins;
    return (min + delta * (binid + 1));
  } else {
    return quantile[colselector[colid] * nbins + binid];
  }
}
