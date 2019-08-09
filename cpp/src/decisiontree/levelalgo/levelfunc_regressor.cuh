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
#include "common_helper.cuh"
#include "flatnode.h"
#include "levelhelper_regressor.cuh"
#include "metric.cuh"
/*
This is the driver function for building regression tree 
level by level using a simple for loop.
At each level; following steps are involved.
1. Set up parent node mean and counts
2. Compute means and counts for all nodes, all cols and all bins.
3. Find best split col and bin for each node.
4. Check info gain and then leaf out nodes as needed.
5. make split.
*/
template <typename T>
ML::DecisionTree::TreeNode<T, T>* grow_deep_tree_regression(
  const T* data, const T* labels, unsigned int* rowids,
  const std::vector<unsigned int>& feature_selector, const int n_sampled_rows,
  const int nrows, const int nbins, int maxdepth, const int maxleaves,
  const int min_rows_per_node, const ML::CRITERION split_cr, int split_algo,
  int& depth_cnt, int& leaf_cnt,
  std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
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

  size_t total_nodes = pow(2, (maxdepth + 1)) - 1;

  std::vector<T> sparse_meanstate;
  std::vector<unsigned int> sparse_countstate;
  sparse_meanstate.resize(total_nodes, 0.0);
  sparse_countstate.resize(total_nodes, 0);
  sparse_meanstate[0] = mean;
  sparse_countstate[0] = count;
  std::vector<SparseTreeNode<T, T>> sparsetree;
  sparsetree.reserve(total_nodes);
  SparseTreeNode<T, T> sparsenode;
  sparsenode.best_metric_val = initial_metric;
  sparsetree.push_back(sparsenode);
  int sparsesize = 0;
  int sparsesize_nextitr = 0;

  int n_nodes = 1;
  int n_nodes_nextitr = 1;
  std::vector<int> sparse_nodelist;
  sparse_nodelist.reserve(pow(2, maxdepth));
  sparse_nodelist.push_back(0);

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
    sparsesize = sparsesize_nextitr;
    sparsesize_nextitr = sparsetree.size();

    ASSERT(
      n_nodes <= tempmem->max_nodes_per_level,
      "Max node limit reached. Requested nodes %d > %d max nodes at depth %d\n",
      n_nodes, tempmem->max_nodes_per_level, depth);
    init_parent_value(sparse_meanstate, sparse_countstate, sparse_nodelist,
                      sparsesize, depth, tempmem);

    if (split_cr == ML::CRITERION::MSE) {
      get_mse_regression<T, SquareFunctor>(
        data, labels, flagsptr, sample_cnt, nrows, ncols, nbins, n_nodes,
        split_algo, tempmem, d_mseout, d_predout, d_count);
    } else {
      get_mse_regression<T, AbsFunctor>(
        data, labels, flagsptr, sample_cnt, nrows, ncols, nbins, n_nodes,
        split_algo, tempmem, d_mseout, d_predout, d_count);
    }

    float* infogain = tempmem->h_outgain->data();
    get_best_split_regression(
      h_mseout, d_mseout, h_predout, d_predout, h_count, d_count,
      feature_selector, d_colids, nbins, n_nodes, depth, min_rows_per_node,
      split_algo, sparsesize, infogain, sparse_meanstate, sparse_countstate,
      sparsetree, sparse_nodelist, h_split_colidx, h_split_binidx,
      d_split_colidx, d_split_binidx, tempmem);

    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    leaf_eval_regression(infogain, depth, maxdepth, maxleaves, h_new_node_flags,
                         sparsetree, sparsesize, sparse_meanstate,
                         n_nodes_nextitr, sparse_nodelist, leaf_cnt);

    MLCommon::updateDevice(d_new_node_flags, h_new_node_flags, n_nodes,
                           tempmem->stream);
    make_level_split(data, nrows, ncols, nbins, n_nodes, split_algo,
                     d_split_colidx, d_split_binidx, d_new_node_flags, flagsptr,
                     tempmem);
  }
  for (int i = sparsesize_nextitr; i < sparsetree.size(); i++) {
    sparsetree[i].prediction = sparse_meanstate[i];
  }
  return go_recursive_sparse(sparsetree);
}
