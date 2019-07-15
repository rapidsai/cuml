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
void setup_sampling(unsigned int *flagsptr, unsigned int *sample_cnt,
                    const unsigned int *rowids, const int nrows,
                    const int n_sampled_rows, cudaStream_t &stream) {
  CUDA_CHECK(cudaMemsetAsync(sample_cnt, 0, nrows * sizeof(int), stream));
  int threads = 256;
  int blocks = MLCommon::ceildiv(n_sampled_rows, threads);
  if (blocks > 65536) blocks = 65536;
  setup_counts_kernel<<<blocks, threads, 0, stream>>>(sample_cnt, rowids,
                                                      n_sampled_rows);
  CUDA_CHECK(cudaGetLastError());
  blocks = MLCommon::ceildiv(nrows, threads);
  if (blocks > 65536) blocks = 65536;
  setup_flags_kernel<<<blocks, threads, 0, stream>>>(sample_cnt, flagsptr,
                                                     nrows);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename L>
void make_level_split(T *data, const int nrows, const int ncols,
                      const int nbins, const int n_nodes, int *split_colidx,
                      int *split_binidx, const unsigned int *new_node_flags,
                      unsigned int *flags,
                      std::shared_ptr<TemporaryMemory<T, L>> tempmem) {
  int threads = 256;
  int blocks = MLCommon::ceildiv(nrows, threads);
  split_level_kernel<<<blocks, threads, 0, tempmem->stream>>>(
    data, tempmem->d_quantile->data(), split_colidx, split_binidx, nrows, ncols,
    nbins, n_nodes, new_node_flags, flags);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename L>
ML::DecisionTree::TreeNode<T, L> *go_recursive(
  std::vector<FlatTreeNode<T, L>> &flattree, int idx = 0) {
  ML::DecisionTree::TreeNode<T, L> *node = NULL;
  if (idx < flattree.size()) {
    node = new ML::DecisionTree::TreeNode<T, L>();
    node->split_metric_val = flattree[idx].best_metric_val;
    node->question.column = flattree[idx].colid;
    node->question.value = flattree[idx].quesval;
    node->prediction = flattree[idx].prediction;
    if (flattree[idx].type == true) {
      return node;
    }
    node->left = go_recursive(flattree, 2 * idx + 1);
    node->right = go_recursive(flattree, 2 * idx + 2);
  }
  return node;
}
