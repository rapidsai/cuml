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
#include "../kernels/metric.cuh"
#include "../kernels/metric_def.h"
#include "common_helper.cuh"
#include "flatnode.h"
#include "levelhelper_classifier.cuh"

template <typename T>
ML::DecisionTree::TreeNode<T, int>* grow_deep_tree_classification(
  const ML::cumlHandle_impl& handle, T* data, int* labels, unsigned int* rowids,
  const std::vector<unsigned int>& feature_selector, int n_sampled_rows,
  const int nrows, const int n_unique_labels, const int nbins,
  const int maxdepth, const int maxleaves, const int min_rows_per_node,
  const ML::CRITERION split_cr, int& depth_cnt, int& leaf_cnt,
  std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  const int ncols = feature_selector.size();
  MLCommon::updateDevice(tempmem->d_colids->data(), feature_selector.data(),
                         feature_selector.size(), tempmem->stream);

  unsigned int* flagsptr = tempmem->d_flags->data();
  unsigned int* sample_cnt = tempmem->d_sample_cnt->data();
  setup_sampling(flagsptr, sample_cnt, rowids, nrows, n_sampled_rows,
                 tempmem->stream);
  std::vector<int> histvec;
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
  size_t total_nodes = 0;
  for (int i = 0; i <= maxdepth; i++) {
    total_nodes += pow(2, i);
  }
  std::vector<std::vector<int>> histstate;
  histstate.resize(total_nodes);
  for (int i = 0; i < total_nodes; i++) {
    std::vector<int> tmp(n_unique_labels, 0);
    histstate[i] = tmp;
  }
  histstate[0] = histvec;
  std::vector<FlatTreeNode<T, int>> flattree;
  flattree.resize(total_nodes);
  FlatTreeNode<T, int> node;
  node.best_metric_val = initial_metric;
  flattree[0] = node;
  int n_nodes = 1;
  int n_nodes_nextitr = 1;
  std::vector<int> nodelist;
  nodelist.push_back(0);
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
    get_histogram_classification(data, labels, flagsptr, sample_cnt, nrows,
                                 ncols, n_unique_labels, nbins, n_nodes,
                                 tempmem, d_histogram);

    std::vector<float> infogain;
    if (split_cr == ML::CRITERION::GINI) {
      get_best_split_classification<T, GiniFunctor, GiniDevFunctor>(
        h_histogram, d_histogram, feature_selector, d_colids, nbins,
        n_unique_labels, n_nodes, depth, min_rows_per_node, infogain, histstate,
        flattree, nodelist, h_split_colidx, h_split_binidx, d_split_colidx,
        d_split_binidx, tempmem);
    } else {
      get_best_split_classification<T, EntropyFunctor, EntropyDevFunctor>(
        h_histogram, d_histogram, feature_selector, d_colids, nbins,
        n_unique_labels, n_nodes, depth, min_rows_per_node, infogain, histstate,
        flattree, nodelist, h_split_colidx, h_split_binidx, d_split_colidx,
        d_split_binidx, tempmem);
    }

    leaf_eval_classification(infogain, depth, maxdepth, maxleaves,
                             h_new_node_flags, flattree, histstate,
                             n_nodes_nextitr, nodelist, leaf_cnt);

    MLCommon::updateDevice(d_new_node_flags, h_new_node_flags, n_nodes,
                           tempmem->stream);

    make_level_split(data, nrows, ncols, nbins, n_nodes, d_split_colidx,
                     d_split_binidx, d_new_node_flags, flagsptr, tempmem);

    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  }
  int nleaves = pow(2, maxdepth);
  int leaf_st = flattree.size() - nleaves;
  for (int i = 0; i < nleaves; i++) {
    flattree[leaf_st + i].prediction = get_class_hist(histstate[leaf_st + i]);
  }
  return go_recursive<T,int>(flattree);
}
