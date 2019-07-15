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
  unsigned int count;
  if (split_cr == ML::CRITERION::MSE) {
    initial_metric_regression<T, SquareFunctor>(labels, sample_cnt, nrows, mean,
                                                count, initial_metric, tempmem);
  } else {
    initial_metric_regression<T, AbsFunctor>(labels, sample_cnt, nrows, mean,
                                             count, initial_metric, tempmem);
  }
  size_t total_nodes = 0;
  for (int i = 0; i <= maxdepth; i++) {
    total_nodes += pow(2, i);
  }
  std::vector<T> meanstate;
  std::vector<unsigned int> countstate;
  meanstate.resize(total_nodes, 0.0);
  countstate.resize(total_nodes, 0);
  meanstate[0] = mean;
  countstate[0] = count;
  std::vector<FlatTreeNode<T, T>> flattree;
  flattree.resize(total_nodes);
  FlatTreeNode<T, T> node;
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
  unsigned int* h_count = tempmem->h_count->data();
  unsigned int* d_count = tempmem->d_count->data();
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
    if (split_cr == ML::CRITERION::MSE) {
      get_mse_regression<T, SquareFunctor>(
        data, labels, flagsptr, sample_cnt, nrows, ncols, nbins, n_nodes,
        tempmem, d_mseout, d_predout, d_count);
    } else {
      get_mse_regression<T, AbsFunctor>(data, labels, flagsptr, sample_cnt,
                                        nrows, ncols, nbins, n_nodes, tempmem,
                                        d_mseout, d_predout, d_count);
    }

    std::vector<float> infogain;
    get_best_split_regression(
      h_mseout, d_mseout, h_predout, d_predout, h_count, d_count,
      feature_selector, d_colids, nbins, n_nodes, depth, min_rows_per_node,
      infogain, meanstate, countstate, flattree, nodelist, h_split_colidx,
      h_split_binidx, d_split_colidx, d_split_binidx, tempmem);

    leaf_eval_regression(infogain, depth, maxdepth, maxleaves, h_new_node_flags,
                         flattree, meanstate, countstate, n_nodes_nextitr,
                         nodelist, leaf_cnt);

    MLCommon::updateDevice(d_new_node_flags, h_new_node_flags, n_nodes,
                           tempmem->stream);
    make_level_split(data, nrows, ncols, nbins, n_nodes, d_split_colidx,
                     d_split_binidx, d_new_node_flags, flagsptr, tempmem);

    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    break;
  }
  int nleaves = pow(2, maxdepth);
  int leaf_st = flattree.size() - nleaves;
  for (int i = 0; i < nleaves; i++) {
    flattree[leaf_st + i].prediction = meanstate[leaf_st + i];
  }
  return go_recursive<T, T>(flattree);
}
