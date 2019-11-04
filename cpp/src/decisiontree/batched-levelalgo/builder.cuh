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

#include <cuda_utils.h>
#include <cuml/tree/decisiontree.hpp>
#include "input.cuh"
#include "node.cuh"
#include "split.cuh"

namespace ML {
namespace DecisionTree {

/**
 * Internal struct used to do all the heavy-lifting required for tree building
 *
 * @note This struct does NOT own any of the underlying device/host pointers.
 *       They all must explicitly be allocated by the caller and passed to it.
 */
template <typename DataT, typename LabelT, typename IdxT>
struct Builder {
  typedef Node<DataT, LabelT, IdxT> NodeT;
  typedef Split<DataT, IdxT> SplitT;
  typedef Input<DataT, LabelT, IdxT> InputT;

  /** DT params */
  DecisionTreeParams params;
  /** input dataset */
  InputT input;

  /** max nodes that we can create */
  IdxT maxNodes;
  /** total number of histogram bins */
  IdxT nHistBins;
  /** gain/metric before splitting root node */
  DataT rootGain;

  /** number of nodes created in the current batch */
  IdxT* n_nodes;
  /** class histograms */
  int* hist;
  /** threadblock arrival count */
  int* done_count;
  /** mutex array used for atomically updating best split */
  int* mutex;
  /** number of leaves created so far */
  volatile IdxT* n_leaves;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  NodeT* curr_nodes;
  /** next batch of nodes */
  NodeT* next_nodes;

  /** host copy of the number of new nodes in current branch */
  IdxT* h_new_n_nodes;
  /** host copy for initial histograms */
  int* h_hist;
  /** list of nodes (must be allocated using cudaMallocHost!) */
  NodeT* h_nodes;
  /** list of splits (must be allocated using cudaMallocHost!) */
  SplitT* h_splits;
  /** total number of nodes created so far */
  IdxT h_n_nodes;
  /** range of the currently worked upon nodes */
  IdxT node_start, node_end;


  /** checks if this struct is being used for classification or regression */
  static constexpr bool isRegression() {
    return std::is_same<DataT, LabelT>::value;
  }

  /**
   * @brief Computes workspace size needed for the current computation
   * @param d_wsize (in B) of the device workspace to be allocated
   * @param h_wsize (in B) of the host workspace to be allocated
   * @param p the input params
   * @param data input col-major dataset on device (dim = totalRows x totalCols)
   * @param labels output label for each row in the dataset (len = totalRows)
   *               It should be on device.
   * @param totalRows total rows in the dataset
   * @param totalCols total cols in the dataset
   * @param sampledRows number of rows sampled in the dataset
   * @param sampledCols number of cols sampled in the dataset
   * @param rowids sampled row ids (on device) (len = sampledRows)
   * @param colids sampled col ids (on device) (len = sampledCols)
   * @param nclasses number of output classes (only for classification)
   * @param quantiles histogram/quantile bins of the input dataset, for each of
   *                  its column. Pass a nullptr if this needs to be computed
   *                  fresh. (on device, col-major) (dim = nbins x sampledCols)
   */
  void workspaceSize(size_t& d_wsize, size_t& h_wsize,
                     const DecisionTreeParams& p, DataT* data, LabelT* labels,
                     IdxT totalRows, IdxT totalCols, IdxT sampledRows,
                     IdxT sampledCols, IdxT* rowids, IdxT* colids,
                     IdxT nclasses, DataT* quantiles) {
    ASSERT(!isRegression(), "Currently only classification is supported!");
    ASSERT(quantiles != nullptr,
           "Currently quantiles need to be computed before this call!");
    params = p;
    params.n_blks_for_cols = std::min(nSampledCols, p.n_blks_for_cols);
    input.data = data;
    input.labels = labels;
    input.M = totalRows;
    input.N = totalCols;
    input.nSampledRows = nSampledRows;
    input.nSampledCols = nSampledCols;
    input.rowids = rowids;
    input.colids = colids;
    input.nclasses = nclasses;
    input.quantiles = quantiles;
    auto max_batch = params.max_batch_size;
    auto n_col_blks = params.n_blks_for_cols;
    nHistBins = 2 * max_batch * (params.n_bins + 1) * n_col_blks;
    // x3 just to be safe since we can't strictly adhere to max_leaves
    maxNodes = params.max_leaves * 3;
    d_wsize = 0;
    d_wsize += sizeof(IdxT);                          // n_nodes
    d_wsize += sizeof(int) * nHistBins;               // hist
    d_wsize += sizeof(int) * max_batch * n_col_blks;  // done_count
    d_wsize += sizeof(int) * max_batch;               // mutex
    d_wsize += sizeof(IdxT);                          // n_leaves
    d_wsize += sizeof(SplitT) * max_batch;            // splits
    d_wsize += sizeof(NodeT) * max_batch;             // curr_nodes
    d_wsize += sizeof(NodeT) * 2 * max_batch;         // next_nodes
    // all nodes in the tree
    h_wsize = sizeof(IdxT);                   // h_new_n_nodes
    h_wsize += sizeof(int) * input.nclasses;  // h_hist
    h_wsize += sizeof(NodeT) * maxNodes;      // h_nodes
    h_wsize += sizeof(SplitT) * maxNodes;     // h_splits
  }

  /**
   * @brief assign workspace to the current state
   * @param d_wspace device buffer allocated by the user for the workspace. Its
   *                 size should be atleast workspaceSize()
   * @param h_wspace pinned host buffer mainly needed to store the learned nodes
   */
  void assignWorkspace(char* d_wspace, char* h_wspace) {
    auto max_batch = params.max_batch_size;
    auto n_col_blks = params.n_blks_for_cols;
    // device
    n_nodes = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT);
    hist = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * nHistBins;
    done_count = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * max_batch * n_col_blks;
    mutex = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * max_batch;
    n_leaves = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT);
    splits = reinterpret_cast<SplitT*>(d_wspace);
    d_wspace += sizeof(SplitT) * max_batch;
    curr_nodes = reinterpret_cast<NodeT*>(d_wspace);
    d_wspace += sizeof(NodeT) * max_batch;
    next_nodes = reinterpret_cast<NodeT*>(d_wspace);
    // host
    h_new_n_nodes = reinterpret_cast<IdxT*>(h_wspace);
    h_wspace += sizeof(IdxT);
    h_hist = reinterpret_cast<int*>(h_wspace);
    h_wspace += sizeof(int) * input.nclasses;
    h_nodes = reinterpret_cast<NodeT*>(h_wspace);
    h_wspace += sizeof(NodeT) * maxNodes;
    h_splits = reinterpret_cast<Split*>(h_wspace);
  }

  /** Main training method. To be called only after `assignWorkspace()` */
  void train(cudaStream_t s) {
    init(s);
    while (!isOver()) {
      auto new_nodes = doSplit(s);
      h_n_nodes += new_nodes;
      updateNodeRange();
    }
  }

 private:
  ///@todo: support starting from arbitrary nodes
  /**
   * @brief Initialize buffers and state
   * @param s cuda stream
   */
  void init(cudaStream_t s) {
    auto max_batch = params.max_batch_size;
    auto n_col_blks = params.n_blks_for_cols;
    CUDA_CHECK(cudaMemsetAsync(done_count, 0,
                               sizeof(int) * max_batch * n_col_blks, s));
    CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, s));
    CUDA_CHECK(cudaMemsetAsync(n_leaves, 0, sizeof(IdxT), s));
    rootGain = initialMetric(s);
    node_start = 0;
    node_end = h_n_nodes = 1;  // start with root node
    h_nodes[0].parentGain = rootGain;
    h_nodes[0].start = 0;
    h_nodes[0].end = input.nSampledRows;
    h_nodes[0].depth = 0;
  }

  /** check whether any more nodes need to be processed or not */
  bool isOver() const { return node_end == h_n_nodes; }

  /**
   * @brief After the current batch is finished processing, update the range
   *        of nodes to be worked upon in the next batch
   */
  void updateNodeRange() {
    node_start = node_end;
    auto nodes_remaining = h_n_nodes - node_end;
    node_end = std::min(nodes_remaining, params.max_batch_size) + node_end;
  }

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 256;
  /** threads per block for the nodeSplitKernel */
  static constexpr int TPB_SPLIT = 512;

  /**
   * Computes best split across all nodes in the current batch and splits the
   * nodes accordingly
   * @param s cuda stream
   * @return the number of newly created nodes
   */
  IdxT doSplit(cudaStream_t s) {
    auto nbins = params.n_bins;
    auto nclasses = input.nclasses;
    auto binSize = nbins * 2 * nclasses;
    auto len = binSize + 2 * nbins;
    size_t smemSize = sizeof(int) * len + sizeof(DataT) * nbins;
    auto batchSize = node_end - node_start;
    auto nblks = MLCommon::ceildiv<int>(batchSize, TPB_DEFAULT);
    // start fresh on the number of *new* nodes created in this batch
    CUDA_CHECK(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), s));
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, batchSize, s);
    // get the current set of nodes to be worked upon
    MLCommon::updateDevice(curr_nodes, h_nodes + node_start, batchSize, s);
    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    auto n_col_blks = params.n_blks_for_cols;
    dim3 grid(params.n_blks_for_rows, n_col_blks, batchSize);
    for (IdxT c = 0; c < ncols; c += n_col_blks) {
      CUDA_CHECK(cudaMemsetAsync(hist, 0, sizeof(int) * nHistBins, s));
      computeSplitKernel<DataT, LabelT, SplitT, TPB_DEFAULT>
        <<<grid, TPB_DEFAULT, smemSize, s>>>(
          hist, params, input, curr_nodes, c, done_count, mutex, n_leaves,
          rowids, splits, ncols, colids, quantiles);
      CUDA_CHECK(cudaGetLastError());
    }
    // create child nodes (or make the current ones leaf)
    smemSize = std::max(2 * sizeof(IdxT) * TPB_SPLIT, sizeof(int) * nclasses);
    nodeSplitKernel<DataT, LabelT, IdxT, TPB_SPLIT>
      <<<batchSize, TPB_SPLIT, smemSize, s>>>(params, input, curr_nodes,
                                              next_nodes, n_nodes, rowids,
                                              splits, n_leaves, h_n_nodes);
    CUDA_CHECK(cudaGetLastError());
    // copy the best splits to host
    MLCommon::updateHost(h_splits + node_start, splits, batchSize, s);
    // copy the updated (due to leaf creation) and newly created child nodes
    MLCommon::updateHost(h_nodes + node_start, curr_nodes, batchSize, s);
    MLCommon::updateHost(h_new_n_nodes, n_nodes, 1, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    MLCommon::updateHost(h_nodes + h_n_nodes, next_nodes, *h_new_n_nodes, s);
    return *h_new_n_nodes;
  }

  /** computes the initial metric needed for root node split decision */
  DataT initialMetric(cudaStream_t s) {
    static constexpr int TPB = 256;
    static constexpr int NITEMS = 8;
    int nblks = ceildiv(nSampledRows, TPB * NITEMS);
    size_t smemSize = sizeof(int) * input.nclasses;
    auto out = DataT(1.0);
    ///@todo: support for regression
    if (isRegression()) {
    } else {
      // reusing `hist` for initial bin computation only
      CUDA_CHECK(cudaMemsetAsync(hist, 0, sizeof(int) * input.nclasses, s));
      initialClassHistKernel<DataT, LabelT, IdxT><<<nblks, TPB, smemSize, s>>>(
        hist, rowids, input.labels, input.nclasses, nSampledRows);
      CUDA_CHECK(cudaGetLastError());
      MLCommon::updateHost(h_hist, hist, input.nclasses, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      // better to compute the initial metric (after class histograms) on CPU
      ///@todo: support other metrics
      auto invlen = out / DataT(nSampledRows);
      for (IdxT i = 0; i < input.nclasses; ++i) {
        auto val = h_hist[i] * invlen;
        out -= val * val;
      }
    }
    return out;
  }
};  // end Builder

}  // namespace DecisionTree
}  // namespace ML
