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
  const T* data, const int* labels, unsigned int* rowids, const int Ncols,
  const float colper, int n_sampled_rows, const int nrows,
  const int n_unique_labels, const int nbins, const int maxdepth,
  const int maxleaves, const int min_rows_per_node,
  const ML::CRITERION split_cr, const int split_algo,
  const float min_impurity_decrease, int& depth_cnt, int& leaf_cnt,
  std::vector<SparseTreeNode<T, int>>& sparsetree, const int treeid,
  std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  const int ncols_sampled = (int)(colper * Ncols);
  unsigned int* flagsptr = tempmem->d_flags->data();
  unsigned int* sample_cnt = tempmem->d_sample_cnt->data();
  setup_sampling(flagsptr, sample_cnt, rowids, nrows, n_sampled_rows,
                 tempmem->stream);
  std::vector<unsigned int> histvec(n_unique_labels, 0);
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
  int reserve_depth = std::min(tempmem->swap_depth, maxdepth);
  size_t total_nodes = pow(2, (reserve_depth + 1)) - 1;

  unsigned int* h_parent_hist = tempmem->h_parent_hist->data();
  unsigned int* h_child_hist = tempmem->h_child_hist->data();
  memcpy(h_parent_hist, histvec.data(), n_unique_labels * sizeof(int));

  sparsetree.reserve(total_nodes);
  SparseTreeNode<T, int> sparsenode;
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
  std::uniform_int_distribution<unsigned int> dist(0, Ncols - 1);
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
  for (int depth = 0; (depth < tempmem->swap_depth) && (n_nodes_nextitr != 0);
       depth++) {
    depth_cnt = depth + 1;
    n_nodes = n_nodes_nextitr;
    sparsesize = sparsesize_nextitr;
    sparsesize_nextitr = sparsetree.size();
    ASSERT(
      n_nodes <= tempmem->max_nodes_per_level,
      "Max node limit reached. Requested nodes %d > %d max nodes at depth %d\n",
      n_nodes, tempmem->max_nodes_per_level, depth);

    update_feature_sampling(h_colids, d_colids, h_colstart, d_colstart, Ncols,
                            ncols_sampled, n_nodes, mtg, dist, feature_selector,
                            tempmem, d_rng);
    get_histogram_classification(data, labels, flagsptr, sample_cnt, nrows,
                                 Ncols, ncols_sampled, n_unique_labels, nbins,
                                 n_nodes, split_algo, tempmem, d_histogram);

    float* infogain = tempmem->h_outgain->data();
    if (split_cr == ML::CRITERION::GINI) {
      get_best_split_classification<T, GiniFunctor, GiniDevFunctor>(
        h_histogram, d_histogram, h_colids, d_colids, h_colstart, d_colstart,
        Ncols, ncols_sampled, nbins, n_unique_labels, n_nodes, depth,
        min_rows_per_node, split_algo, infogain, h_parent_hist, h_child_hist,
        sparsetree, sparsesize, sparse_nodelist, h_split_colidx, h_split_binidx,
        d_split_colidx, d_split_binidx, tempmem);
    } else {
      get_best_split_classification<T, EntropyFunctor, EntropyDevFunctor>(
        h_histogram, d_histogram, h_colids, d_colids, h_colstart, d_colstart,
        Ncols, ncols_sampled, nbins, n_unique_labels, n_nodes, depth,
        min_rows_per_node, split_algo, infogain, h_parent_hist, h_child_hist,
        sparsetree, sparsesize, sparse_nodelist, h_split_colidx, h_split_binidx,
        d_split_colidx, d_split_binidx, tempmem);
    }

    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    leaf_eval_classification(infogain, depth, min_impurity_decrease, maxdepth,
                             n_unique_labels, maxleaves, h_new_node_flags,
                             sparsetree, sparsesize, h_parent_hist,
                             n_nodes_nextitr, sparse_nodelist, leaf_cnt);

    MLCommon::updateDevice(d_new_node_flags, h_new_node_flags, n_nodes,
                           tempmem->stream);
    make_level_split(data, nrows, Ncols, ncols_sampled, nbins, n_nodes,
                     split_algo, d_split_colidx, d_split_binidx,
                     d_new_node_flags, flagsptr, tempmem);

    memcpy(h_parent_hist, h_child_hist,
           2 * n_nodes * n_unique_labels * sizeof(unsigned int));
  }
  // Start of gather algorithm
  //Convertor
  //std::cout << "begin gather \n";
  int lastsize = sparsetree.size() - sparsesize_nextitr;
  n_nodes = n_nodes_nextitr;
  if (n_nodes == 0) return;
  unsigned int *d_nodecount, *d_samplelist, *d_nodestart;
  SparseTreeNode<T, int>* d_sparsenodes;
  SparseTreeNode<T, int>* h_sparsenodes;
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

    if (split_cr == ML::CRITERION::GINI) {
      best_split_gather_classification<T, GiniDevFunctor>(
        data, labels, d_colids, d_colstart, d_nodestart, d_samplelist, nrows,
        Ncols, ncols_sampled, n_unique_labels, nbins, n_nodes, split_algo,
        sparsetree.size() + lastsize, min_impurity_decrease, tempmem,
        d_sparsenodes, d_nodelist);
    } else {
      best_split_gather_classification<T, EntropyDevFunctor>(
        data, labels, d_colids, d_colstart, d_nodestart, d_samplelist, nrows,
        Ncols, ncols_sampled, n_unique_labels, nbins, n_nodes, split_algo,
        sparsetree.size() + lastsize, min_impurity_decrease, tempmem,
        d_sparsenodes, d_nodelist);
    }
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
    if (split_cr == ML::CRITERION::GINI) {
      make_leaf_gather_classification<T, GiniDevFunctor>(
        labels, d_nodestart, d_samplelist, n_unique_labels, d_sparsenodes,
        d_nodelist, n_nodes, tempmem);
    } else {
      make_leaf_gather_classification<T, EntropyDevFunctor>(
        labels, d_nodestart, d_samplelist, n_unique_labels, d_sparsenodes,
        d_nodelist, n_nodes, tempmem);
    }
    MLCommon::updateHost(h_sparsenodes, d_sparsenodes, lastsize,
                         tempmem->stream);
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    sparsetree.insert(sparsetree.end(), h_sparsenodes,
                      h_sparsenodes + lastsize);
  }
}
