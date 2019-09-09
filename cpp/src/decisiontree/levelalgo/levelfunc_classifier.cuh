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
#include <iostream>
#include <numeric>
#include "../decisiontree.hpp"
#include "../flatnode.h"
#include "common_helper.cuh"
#include "levelhelper_classifier.cuh"
#include "metric.cuh"

/*
This is the driver function for building classification tree 
level by level using a simple for loop.
At each level; following steps are involved.
1. Compute histograms for all nodes, all cols and all bins.
2. Find best split col and bin for each node.
3. Check info gain and then leaf out nodes as needed.
4. make split.
*/
template <typename T>
void grow_deep_tree_classification(
  const T* data, const int* labels, unsigned int* rowids,
  const std::vector<unsigned int>& feature_selector, int n_sampled_rows,
  const int nrows, const int n_unique_labels, const int nbins,
  const int maxdepth, const int maxleaves, const int min_rows_per_node,
  const ML::CRITERION split_cr, const int split_algo, int& depth_cnt,
  int& leaf_cnt, std::vector<SparseTreeNode<T, int>>& sparsetree,
  std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  const int ncols = feature_selector.size();
  MLCommon::updateDevice(tempmem->d_colids->data(), feature_selector.data(),
                         feature_selector.size(), tempmem->stream);
  unsigned int* flagsptr = tempmem->d_flags->data();
  unsigned int* sample_cnt = tempmem->d_sample_cnt->data();
  setup_sampling(flagsptr, sample_cnt, rowids, nrows, n_sampled_rows,
                 tempmem->stream);
  std::vector<int> histvec(n_unique_labels, 0);
  T initial_metric;
  if (split_cr == ML::CRITERION::GINI) {
    initial_metric_classification<T, GiniFunctor>(labels, sample_cnt, nrows,
                                                  n_unique_labels, histvec,
                                                  initial_metric, tempmem);
  } else {
    initial_metric_classification<T, EntropyFunctor>(labels, sample_cnt, nrows,
                                                     n_unique_labels, histvec,
                                                     initial_metric, tempmem);
  }
  size_t total_nodes = pow(2, (maxdepth + 1)) - 1;

  std::vector<std::vector<int>> sparse_histstate;
  sparse_histstate.resize(total_nodes, std::vector<int>(n_unique_labels));
  sparse_histstate[0] = histvec;

  sparsetree.reserve(total_nodes);
  SparseTreeNode<T, int> sparsenode;
  sparsenode.best_metric_val = initial_metric;
  sparsetree.push_back(sparsenode);
  int sparsesize = 0;
  int sparsesize_nextitr = 0;

  int n_nodes = 1;
  int n_nodes_nextitr = 1;
  std::vector<int> sparse_nodelist;
  sparse_nodelist.reserve(pow(2, maxdepth));
  sparse_nodelist.push_back(0);
  //this can be depth loop

  //Setup pointers
  unsigned int* d_histogram = tempmem->d_histogram->data();
  unsigned int* h_histogram = tempmem->h_histogram->data();
  int* h_split_binidx = tempmem->h_split_binidx->data();
  int* d_split_binidx = tempmem->d_split_binidx->data();
  int* h_split_colidx = tempmem->h_split_colidx->data();
  int* d_split_colidx = tempmem->d_split_colidx->data();
  unsigned int* h_new_node_flags = tempmem->h_new_node_flags->data();
  unsigned int* d_new_node_flags = tempmem->d_new_node_flags->data();
  unsigned int* d_colids = tempmem->d_colids->data();

  for (int depth = 0; (depth < maxdepth) && (n_nodes_nextitr != 0); depth++) {
    depth_cnt = depth + 1;
    n_nodes = n_nodes_nextitr;
    sparsesize = sparsesize_nextitr;
    sparsesize_nextitr = sparsetree.size();
    ASSERT(
      n_nodes <= tempmem->max_nodes_per_level,
      "Max node limit reached. Requested nodes %d > %d max nodes at depth %d\n",
      n_nodes, tempmem->max_nodes_per_level, depth);

    get_histogram_classification(data, labels, flagsptr, sample_cnt, nrows,
                                 ncols, n_unique_labels, nbins, n_nodes,
                                 split_algo, tempmem, d_histogram);

    float* infogain = tempmem->h_outgain->data();
    if (split_cr == ML::CRITERION::GINI) {
      get_best_split_classification<T, GiniFunctor, GiniDevFunctor>(
        h_histogram, d_histogram, feature_selector, d_colids, nbins,
        n_unique_labels, n_nodes, depth, min_rows_per_node, split_algo,
        infogain, sparse_histstate, sparsetree, sparsesize, sparse_nodelist,
        h_split_colidx, h_split_binidx, d_split_colidx, d_split_binidx,
        tempmem);
    } else {
      get_best_split_classification<T, EntropyFunctor, EntropyDevFunctor>(
        h_histogram, d_histogram, feature_selector, d_colids, nbins,
        n_unique_labels, n_nodes, depth, min_rows_per_node, split_algo,
        infogain, sparse_histstate, sparsetree, sparsesize, sparse_nodelist,
        h_split_colidx, h_split_binidx, d_split_colidx, d_split_binidx,
        tempmem);
    }

    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

    leaf_eval_classification(
      infogain, depth, maxdepth, maxleaves, h_new_node_flags, sparsetree,
      sparsesize, sparse_histstate, n_nodes_nextitr, sparse_nodelist, leaf_cnt);

    MLCommon::updateDevice(d_new_node_flags, h_new_node_flags, n_nodes,
                           tempmem->stream);
    make_level_split(data, nrows, ncols, nbins, n_nodes, split_algo,
                     d_split_colidx, d_split_binidx, d_new_node_flags, flagsptr,
                     tempmem);
  }
  for (int i = sparsesize_nextitr; i < sparsetree.size(); i++) {
    sparsetree[i].prediction = get_class_hist(sparse_histstate[i]);
  }
}
