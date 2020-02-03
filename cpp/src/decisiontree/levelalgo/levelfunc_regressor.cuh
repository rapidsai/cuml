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
#include <cuml/tree/decisiontree.hpp>
#include <iostream>
#include <numeric>
#include "common_helper.cuh"
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
void grow_deep_tree_regression(
  const T* data, const T* labels, unsigned int* rowids, const int Ncols,
  const float colper, const int n_sampled_rows, const int nrows,
  const int nbins, int maxdepth, const int maxleaves,
  const int min_rows_per_node, const ML::CRITERION split_cr, int split_algo,
  const float min_impurity_decrease, int& depth_cnt, int& leaf_cnt,
  std::vector<SparseTreeNode<T, T>>& sparsetree, const int treeid,
  std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  const int ncols_sampled = (int)(colper * Ncols);
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
  int reserve_depth = std::min(tempmem->swap_depth, maxdepth);
  size_t total_nodes = pow(2, (reserve_depth + 1)) - 1;

  std::vector<T> sparse_meanstate;
  std::vector<unsigned int> sparse_countstate;
  sparse_meanstate.resize(total_nodes, 0.0);
  sparse_countstate.resize(total_nodes, 0);
  sparse_meanstate[0] = mean;
  sparse_countstate[0] = count;

  sparsetree.reserve(total_nodes);
  SparseTreeNode<T, T> sparsenode;
  sparsenode.best_metric_val = initial_metric;
  sparsetree.push_back(sparsenode);
  int sparsesize = 0;
  int sparsesize_nextitr = 0;

  int n_nodes = 1;
  int n_nodes_nextitr = 1;
  std::vector<int> sparse_nodelist;
  sparse_nodelist.reserve(tempmem->max_nodes_per_level);
  sparse_nodelist.push_back(0);
  //RNG setup
  std::mt19937 mtg(treeid * 1000);
  MLCommon::Random::Rng d_rng(treeid * 1000);
  std::uniform_int_distribution<int> dist(0, Ncols - 1);

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
  unsigned int* h_colids = tempmem->h_colids->data();
  unsigned int* d_colstart = nullptr;
  unsigned int* h_colstart = nullptr;
  if (tempmem->d_colstart != nullptr) {
    d_colstart = tempmem->d_colstart->data();
    h_colstart = tempmem->h_colstart->data();
    CUDA_CHECK(cudaMemsetAsync(
      d_colstart, 0, tempmem->max_nodes_per_level * sizeof(unsigned int),
      tempmem->stream));
    memset(h_colstart, 0, tempmem->max_nodes_per_level * sizeof(unsigned int));
    MLCommon::updateDevice(d_colids, h_colids, Ncols, tempmem->stream);
  }
  std::vector<unsigned int> feature_selector(h_colids, h_colids + Ncols);
  float* infogain = tempmem->h_outgain->data();

  for (int depth = 0; (depth < tempmem->swap_depth) && (n_nodes_nextitr != 0);
       depth++) {
    depth_cnt = depth + 1;
    n_nodes = n_nodes_nextitr;
    update_feature_sampling(h_colids, d_colids, h_colstart, d_colstart, Ncols,
                            ncols_sampled, n_nodes, mtg, dist, feature_selector,
                            tempmem, d_rng);
    sparsesize = sparsesize_nextitr;
    sparsesize_nextitr = sparsetree.size();

    ASSERT(
      n_nodes <= tempmem->max_nodes_per_level,
      "Max node limit reached. Requested nodes %d > %d max nodes at depth %d\n",
      n_nodes, tempmem->max_nodes_per_level, depth);
    init_parent_value(sparse_meanstate, sparse_countstate, sparse_nodelist,
                      sparsesize, depth, tempmem);

    if (split_cr == ML::CRITERION::MSE) {
      get_mse_regression_fused<T>(
        data, labels, flagsptr, sample_cnt, nrows, Ncols, ncols_sampled, nbins,
        n_nodes, split_algo, tempmem, d_mseout, d_predout, d_count);
      get_best_split_regression<T, MSEGain<T>>(
        h_mseout, d_mseout, h_predout, d_predout, h_count, d_count, h_colids,
        d_colids, h_colstart, d_colstart, Ncols, ncols_sampled, nbins, n_nodes,
        depth, min_rows_per_node, split_algo, sparsesize, infogain,
        sparse_meanstate, sparse_countstate, sparsetree, sparse_nodelist,
        h_split_colidx, h_split_binidx, d_split_colidx, d_split_binidx,
        tempmem);

    } else {
      get_mse_regression<T, AbsFunctor>(
        data, labels, flagsptr, sample_cnt, nrows, Ncols, ncols_sampled, nbins,
        n_nodes, split_algo, tempmem, d_mseout, d_predout, d_count);
      get_best_split_regression<T, MAEGain<T>>(
        h_mseout, d_mseout, h_predout, d_predout, h_count, d_count, h_colids,
        d_colids, h_colstart, d_colstart, Ncols, ncols_sampled, nbins, n_nodes,
        depth, min_rows_per_node, split_algo, sparsesize, infogain,
        sparse_meanstate, sparse_countstate, sparsetree, sparse_nodelist,
        h_split_colidx, h_split_binidx, d_split_colidx, d_split_binidx,
        tempmem);
    }

    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    leaf_eval_regression(infogain, depth, min_impurity_decrease, maxdepth,
                         maxleaves, h_new_node_flags, sparsetree, sparsesize,
                         sparse_meanstate, n_nodes_nextitr, sparse_nodelist,
                         leaf_cnt);

    MLCommon::updateDevice(d_new_node_flags, h_new_node_flags, n_nodes,
                           tempmem->stream);
    make_level_split(data, nrows, Ncols, ncols_sampled, nbins, n_nodes,
                     split_algo, d_split_colidx, d_split_binidx,
                     d_new_node_flags, flagsptr, tempmem);
  }

  // Start of gather algorithm
  //Convertor

  int lastsize = sparsetree.size() - sparsesize_nextitr;
  n_nodes = n_nodes_nextitr;
  if (n_nodes == 0) return;
  unsigned int *d_nodecount, *d_samplelist, *d_nodestart;
  SparseTreeNode<T, T>* d_sparsenodes;
  SparseTreeNode<T, T>* h_sparsenodes;
  int *h_nodelist, *d_nodelist, *d_new_nodelist;
  int max_nodes = tempmem->max_nodes_per_level;
  d_nodecount = (unsigned int*)(tempmem->d_child_best_metric->data());
  d_nodestart = (unsigned int*)(tempmem->d_split_binidx->data());
  d_samplelist = (unsigned int*)(tempmem->d_parent_metric->data());
  d_nodelist = (int*)(tempmem->d_outgain->data());
  d_new_nodelist = (int*)(tempmem->d_split_colidx->data());
  h_nodelist = (int*)(tempmem->h_outgain->data());
  d_sparsenodes = tempmem->d_sparsenodes->data();
  h_sparsenodes = tempmem->h_sparsenodes->data();

  int* h_counter = tempmem->h_counter->data();
  int* d_counter = tempmem->d_counter->data();
  memcpy(h_nodelist, sparse_nodelist.data(),
         sizeof(int) * sparse_nodelist.size());
  MLCommon::updateDevice(d_nodelist, h_nodelist, sparse_nodelist.size(),
                         tempmem->stream);
  //Resize to remove trailing nodes from previous algorithm
  sparsetree.resize(sparsetree.size() - lastsize);
  convert_scatter_to_gather(flagsptr, sample_cnt, n_nodes, nrows, d_nodecount,
                            d_nodestart, d_samplelist, tempmem);
  for (int depth = tempmem->swap_depth; (depth < maxdepth) && (n_nodes != 0);
       depth++) {
    depth_cnt = depth + 1;
    //Algorithm starts here
    update_feature_sampling(h_colids, d_colids, h_colstart, d_colstart, Ncols,
                            ncols_sampled, lastsize, mtg, dist,
                            feature_selector, tempmem, d_rng);

    best_split_gather_regression(
      data, labels, d_colids, d_colstart, d_nodestart, d_samplelist, nrows,
      Ncols, ncols_sampled, nbins, n_nodes, split_algo, split_cr,
      sparsetree.size() + lastsize, min_impurity_decrease, tempmem,
      d_sparsenodes, d_nodelist);

    MLCommon::updateHost(h_sparsenodes, d_sparsenodes, lastsize,
                         tempmem->stream);
    //Update nodelist and split nodes

    make_split_gather(data, d_nodestart, d_samplelist, n_nodes, nrows,
                      d_nodelist, d_new_nodelist, d_nodecount, d_counter,
                      flagsptr, d_sparsenodes, tempmem);
    CUDA_CHECK(cudaMemcpyAsync(d_nodelist, d_new_nodelist,
                               h_counter[0] * sizeof(int),
                               cudaMemcpyDeviceToDevice, tempmem->stream));
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    sparsetree.insert(sparsetree.end(), h_sparsenodes,
                      h_sparsenodes + lastsize);
    lastsize = 2 * n_nodes;
    n_nodes = h_counter[0];
  }
  if (n_nodes != 0) {
    make_leaf_gather_regression(labels, d_nodestart, d_samplelist,
                                d_sparsenodes, d_nodelist, n_nodes, tempmem);
    MLCommon::updateHost(h_sparsenodes, d_sparsenodes, lastsize,
                         tempmem->stream);
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    sparsetree.insert(sparsetree.end(), h_sparsenodes,
                      h_sparsenodes + lastsize);
  }
}
