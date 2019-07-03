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

template <typename T>
ML::DecisionTree::TreeNode<T, int> *grow_deep_tree(
  const ML::cumlHandle_impl &handle, T *data, int *labels, unsigned int *rowids,
  int n_sampled_rows, const int nrows, const int ncols,
  const int n_unique_labels, const int nbins, int maxdepth,
  const std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  std::vector<unsigned int> colselector;
  colselector.resize(ncols);
  std::iota(colselector.begin(), colselector.end(), 0);

  CUDA_CHECK(cudaHostRegister(colselector.data(),
                              sizeof(unsigned int) * colselector.size(),
                              cudaHostRegisterDefault));
  // Copy sampled column IDs to device memory
  MLCommon::updateDevice(tempmem->d_colids->data(), colselector.data(),
                         colselector.size(), tempmem->stream);
  CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  CUDA_CHECK(cudaHostUnregister(colselector.data()));

  MetricInfo<T> split_info;
  gini<T, GiniFunctor>(labels, n_sampled_rows, tempmem, split_info,
                       n_unique_labels);

  MLCommon::device_buffer<unsigned int> *d_flags =
    new MLCommon::device_buffer<unsigned int>(handle.getDeviceAllocator(),
                                              tempmem->stream, nrows);
  unsigned int *flagsptr = d_flags->data();
  CUDA_CHECK(cudaMemsetAsync(flagsptr, 0, nrows * sizeof(unsigned int),
                             tempmem->stream));

  MLCommon::host_buffer<T> *h_quantile = new MLCommon::host_buffer<T>(
    handle.getHostAllocator(), tempmem->stream, nbins * ncols);
  MLCommon::updateHost(h_quantile->data(), tempmem->d_quantile->data(),
                       nbins * ncols, tempmem->stream);

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
  for (int depth = 0; depth < maxdepth; depth++) {
    n_nodes = n_nodes_nextitr;
    /*    std::cout << "number of nodes -->" << n_nodes << std::endl;
    for (int i = 0; i < n_nodes; i++) {
      printf("%d  ", nodelist[i]);
    }
    printf("\n");*/
    size_t histcount = ncols * nbins * n_unique_labels * n_nodes;
    //Allocate all here
    MLCommon::device_buffer<unsigned int> *d_histogram =
      new MLCommon::device_buffer<unsigned int>(handle.getDeviceAllocator(),
                                                tempmem->stream, histcount);
    MLCommon::host_buffer<unsigned int> *h_histogram =
      new MLCommon::host_buffer<unsigned int>(handle.getHostAllocator(),
                                              tempmem->stream, histcount);
    MLCommon::host_buffer<int> *h_split_colidx = new MLCommon::host_buffer<int>(
      handle.getHostAllocator(), tempmem->stream, n_nodes);
    MLCommon::host_buffer<int> *h_split_binidx = new MLCommon::host_buffer<int>(
      handle.getHostAllocator(), tempmem->stream, n_nodes);

    MLCommon::device_buffer<int> *d_split_colidx =
      new MLCommon::device_buffer<int>(handle.getDeviceAllocator(),
                                       tempmem->stream, n_nodes);
    MLCommon::device_buffer<int> *d_split_binidx =
      new MLCommon::device_buffer<int>(handle.getDeviceAllocator(),
                                       tempmem->stream, n_nodes);

    CUDA_CHECK(cudaMemsetAsync(d_histogram->data(), 0,
                               histcount * sizeof(unsigned int),
                               tempmem->stream));
    //End allocation and setups
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
    get_me_histogram(data, labels, flagsptr, nrows, ncols, n_unique_labels,
                     nbins, n_nodes, tempmem, d_histogram->data());

    MLCommon::updateHost(h_histogram->data(), d_histogram->data(), histcount,
                         tempmem->stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    /*    unsigned int *hist = h_histogram->data();
    for (int nid = 0; nid < n_nodes; nid++) {
      for (int j = 0; j < ncols; j++) {
        printf("colid --> %d ;;; ", j);
        for (int i = 0; i < nbins; i++) {
          printf("(%d,%d,%d) ",
                 hist[nid * n_unique_labels * nbins +
                      j * n_nodes * n_unique_labels * nbins + 3 * i],
                 hist[nid * n_unique_labels * nbins +
                      j * n_nodes * n_unique_labels * nbins + 3 * i + 1],
                 hist[nid * n_unique_labels * nbins +
                      j * n_nodes * n_unique_labels * nbins + 3 * i + 2]);
        }
        printf("\n");
      }
    }
    */
    std::vector<float> infogain;
    get_me_best_split<T, GiniFunctor>(
      h_histogram->data(), colselector, nbins, n_unique_labels, n_nodes, depth,
      infogain, histstate, flattree, nodelist, h_split_colidx->data(),
      h_split_binidx->data(), h_quantile->data());

    MLCommon::updateDevice(d_split_binidx->data(), h_split_binidx->data(),
                           n_nodes, tempmem->stream);
    MLCommon::updateDevice(d_split_colidx->data(), h_split_colidx->data(),
                           n_nodes, tempmem->stream);

    MLCommon::host_buffer<unsigned int> *h_new_node_flags =
      new MLCommon::host_buffer<unsigned int>(handle.getHostAllocator(),
                                              tempmem->stream, n_nodes);

    MLCommon::device_buffer<unsigned int> *d_new_node_flags =
      new MLCommon::device_buffer<unsigned int>(handle.getDeviceAllocator(),
                                                tempmem->stream, n_nodes);

    leaf_eval(infogain, depth, maxdepth, h_new_node_flags->data(), flattree,
              histstate, n_nodes_nextitr, nodelist);

    MLCommon::updateDevice(d_new_node_flags->data(), h_new_node_flags->data(),
                           n_nodes, tempmem->stream);

    make_level_split(data, nrows, ncols, nbins, n_nodes, d_split_colidx->data(),
                     d_split_binidx->data(), d_new_node_flags->data(), flagsptr,
                     tempmem);

    //Free
    h_new_node_flags->release(tempmem->stream);
    d_new_node_flags->release(tempmem->stream);
    h_histogram->release(tempmem->stream);
    d_histogram->release(tempmem->stream);
    h_split_colidx->release(tempmem->stream);
    d_split_colidx->release(tempmem->stream);
    h_split_binidx->release(tempmem->stream);
    d_split_binidx->release(tempmem->stream);
    delete h_new_node_flags;
    delete d_new_node_flags;
    delete d_histogram;
    delete h_histogram;
    delete h_split_colidx;
    delete d_split_colidx;
    delete h_split_binidx;
    delete d_split_binidx;
  }
  int nleaves = pow(2, maxdepth);
  int leaf_st = flattree.size() - nleaves;
  for (int i = 0; i < nleaves; i++) {
    flattree[leaf_st + i].prediction = get_class_hist(histstate[leaf_st + i]);
  }
  h_quantile->release(tempmem->stream);
  d_flags->release(tempmem->stream);
  delete d_flags;
  delete h_quantile;
  /*  for (int i = 0; i < flattree.size(); i++) {
    printf("node id--> %d, colid --> %d ques_val --> %f best metric-->%f\n", i,
           flattree[i].colid, flattree[i].quesval, flattree[i].best_metric_val);
	   }*/
  return go_recursive(flattree);
}
