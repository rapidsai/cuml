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
#include "levelhelper_regressor.cuh"

template <typename T>
ML::DecisionTree::TreeNode<T, T>* grow_deep_tree_regression(
  const ML::cumlHandle_impl& handle, T* data, T* labels, unsigned int* rowids,
  const std::vector<unsigned int>& feature_selector, const int n_sampled_rows,
  const int nrows, const int nbins, int maxdepth, const int maxleaves,
  const int min_rows_per_node, const ML::CRITERION split_cr, int& depth_cnt,
  int& leaf_cnt, std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  const int ncols = feature_selector.size();
  MLCommon::updateDevice(tempmem->d_colids->data(), feature_selector.data(),
                         feature_selector.size(), tempmem->stream);

  unsigned int* flagsptr = tempmem->d_flags->data();
  unsigned int* sample_cnt = tempmem->d_sample_cnt->data();
  setup_sampling(flagsptr, sample_cnt, rowids, nrows, n_sampled_rows,
                 tempmem->stream);

  T mean;
  T initial_metric;
  if (split_cr == ML::CRITERION::MSE) {
    initial_metric_regression<T, SquareFunctor>(labels, sample_cnt, nrows, mean,
                                                initial_metric, tempmem);
  } else {
    initial_metric_regression<T, AbsFunctor>(labels, sample_cnt, nrows, mean,
                                             initial_metric, tempmem);
  }
  size_t total_nodes = 0;
  for (int i = 0; i <= maxdepth; i++) {
    total_nodes += pow(2, i);
  }
  std::vector<T> meanstate;
  meanstate.resize(total_nodes, 0.0);
  meanstate[0] = mean;
  std::vector<FlatTreeNode<T>> flattree;
  flattree.resize(total_nodes);
  FlatTreeNode<T> node;
  node.best_metric_val = initial_metric;
  flattree[0] = node;
  int n_nodes = 1;
  int n_nodes_nextitr = 1;
  std::vector<int> nodelist;
  nodelist.push_back(0);
  //this can be depth loop

  //Setup pointers
  T* d_mseout = tempmem->d_mseout->data();
  T* h_mseout = tempmem->h_mseout->data();
  T* d_predout = tempmem->d_predout->data();
  T* h_predout = tempmem->h_predout->data();
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
    get_mse_regression<T, SquareFunctor>(data, labels, flagsptr, sample_cnt,
                                         nrows, ncols, nbins, n_nodes, tempmem,
                                         d_mseout, d_predout);
    break;
  }
  return (new ML::DecisionTree::TreeNode<T, T>());
}
