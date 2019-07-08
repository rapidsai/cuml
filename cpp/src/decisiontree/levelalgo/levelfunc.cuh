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
#include "../decisiontree.h"
#include "../kernels/metric.cuh"
#include "../kernels/metric_def.h"
#include "../memory.cuh"
#include "levelhelper.cuh"
#include "levelkernel.cuh"
#include "levelmem.cuh"

template <typename T>
ML::DecisionTree::TreeNode<T, int>* grow_deep_tree(
  const ML::cumlHandle_impl& handle, T* data, int* labels, unsigned int* rowids,
  int n_sampled_rows, const int nrows, const int ncols,
  const int n_unique_labels, const int nbins, int maxdepth,
  const std::shared_ptr<TemporaryMemory<T, int>> tempmem,
  LevelTemporaryMemory* leveltempmem) {
  std::vector<unsigned int> colselector;
  colselector.resize(ncols);
  std::iota(colselector.begin(), colselector.end(), 0);

  MetricInfo<T> split_info;
  gini<T, GiniFunctor>(labels, n_sampled_rows, tempmem, split_info,
                       n_unique_labels);

  unsigned int* flagsptr = leveltempmem->d_flags->data();
  CUDA_CHECK(cudaMemsetAsync(flagsptr, 0, nrows * sizeof(unsigned int),
                             tempmem->stream));

  std::vector<std::vector<int>> histstate;
  histstate.push_back(split_info.hist);
  std::vector<FlatTreeNode<T>> flattree;
  FlatTreeNode<T> node;
  node.best_metric_val = split_info.best_metric;
  flattree.push_back(node);
  int n_nodes = 1;
  int n_nodes_nextitr = 1;
  std::vector<int> nodelist;
  nodelist.push_back(0);
  //this can be depth loop

  //Setup pointers
  unsigned int* d_histogram = leveltempmem->d_histogram->data();
  unsigned int* h_histogram = leveltempmem->h_histogram->data();
  unsigned int* h_split_binidx = leveltempmem->h_split_binidx->data();
  unsigned int* d_split_binidx = leveltempmem->d_split_binidx->data();
  unsigned int* h_split_colidx = leveltempmem->h_split_colidx->data();
  unsigned int* d_split_colidx = leveltempmem->d_split_colidx->data();
  unsigned int* h_new_node_flags = leveltempmem->h_new_node_flags->data();
  unsigned int* d_new_node_flags = leveltempmem->d_new_node_flags->data();

  for (int depth = 0; depth < maxdepth; depth++) {
    n_nodes = n_nodes_nextitr;
    //End allocation and setups
    get_me_histogram(data, labels, flagsptr, nrows, ncols, n_unique_labels,
                     nbins, n_nodes, leveltempmem->max_nodes, tempmem,
                     d_histogram);

    std::vector<float> infogain;
    get_me_best_split<T, GiniFunctor>(
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
