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
template <class T>
struct FlatTreeNode {
  int prediction = -1;
  int colid = -1;
  T quesval = -99999999999;
  T best_metric_val;
  bool type = false;  // true for leaf node
};

#include <iostream>
#include <numeric>
#include "../decisiontree.hpp"
#include "../kernels/metric.cuh"
#include "../kernels/metric_def.h"
#include "levelhelper.cuh"
#include "levelkernel.cuh"

template <typename T>
ML::DecisionTree::TreeNode<T, int>* grow_deep_tree_classification(
  const ML::cumlHandle_impl& handle, T* data, int* labels, unsigned int* rowids,
  const std::vector<unsigned int>& feature_selector, int n_sampled_rows,
  const int nrows, const int ncols, const int n_unique_labels, const int nbins,
  const int maxdepth, const int maxleaves, const int min_rows_per_node,
  int& depth_cnt, int& leaf_cnt,
  std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  std::vector<unsigned int> colselector;
  colselector.resize(ncols);
  std::iota(colselector.begin(), colselector.end(), 0);

  unsigned int* flagsptr = tempmem->d_flags->data();
  unsigned int* sample_cnt = tempmem->d_sample_cnt->data();
  setup_sampling(flagsptr, sample_cnt, rowids, nrows, n_sampled_rows,
                 tempmem->stream);

  gini_kernel_level<<<MLCommon::ceildiv(nrows, 128), 128,
                      sizeof(int) * n_unique_labels, tempmem->stream>>>(
    labels, sample_cnt, nrows, n_unique_labels,
    (int*)tempmem->d_parent_hist->data());
  CUDA_CHECK(cudaGetLastError());
  MLCommon::updateHost(tempmem->h_parent_hist->data(),
                       tempmem->d_parent_hist->data(), n_unique_labels,
                       tempmem->stream);
  CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  std::vector<int> histvec;
  histvec.assign(tempmem->h_parent_hist->data(),
                 tempmem->h_parent_hist->data() + n_unique_labels);
  T initial_metric = GiniFunctor::exec(histvec, nrows);

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
  unsigned int* d_histogram = tempmem->d_histogram->data();
  unsigned int* h_histogram = tempmem->h_histogram->data();
  int* h_split_binidx = tempmem->h_split_binidx->data();
  int* d_split_binidx = tempmem->d_split_binidx->data();
  int* h_split_colidx = tempmem->h_split_colidx->data();
  int* d_split_colidx = tempmem->d_split_colidx->data();
  unsigned int* h_new_node_flags = tempmem->h_new_node_flags->data();
  unsigned int* d_new_node_flags = tempmem->d_new_node_flags->data();

  for (int depth = 0; depth < maxdepth; depth++) {
    n_nodes = n_nodes_nextitr;
    //End allocation and setups
    get_me_histogram(data, labels, flagsptr, sample_cnt, nrows, ncols,
                     n_unique_labels, nbins, n_nodes, tempmem->max_nodes,
                     tempmem, d_histogram);

    std::vector<float> infogain;
    get_me_best_split<T, GiniFunctor, GiniDevFunctor>(
      h_histogram, d_histogram, colselector, nbins, n_unique_labels, n_nodes,
      depth, infogain, histstate, flattree, nodelist, h_split_colidx,
      h_split_binidx, d_split_colidx, d_split_binidx, tempmem);

    leaf_eval(infogain, depth, maxdepth, h_new_node_flags, flattree, histstate,
              n_nodes_nextitr, nodelist);

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
  return go_recursive(flattree);
}

template <typename T>
ML::DecisionTree::TreeNode<T, T>* grow_deep_tree_regression(
  const ML::cumlHandle_impl& handle, T* data, T* labels, unsigned int* rowids,
  const std::vector<unsigned int>& feature_selector, const int n_sampled_rows,
  const int nrows, const int ncols, const int nbins, int maxdepth,
  const int maxleaves, const int min_rows_per_node, int& depth_cnt,
  int& leaf_cnt, std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  return (new ML::DecisionTree::TreeNode<T, T>());
}
