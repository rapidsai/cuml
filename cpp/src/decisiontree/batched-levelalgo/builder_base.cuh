/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <common/grid_sync.h>
#include <cuda_utils.h>
#include <cuml/tree/flatnode.h>
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>
#include <common/host_buffer.hpp>
#include <cuml/tree/decisiontree.hpp>
#include "input.cuh"
#include "kernels.cuh"
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
template <typename Traits>
struct Builder {
  typedef typename Traits::DataT DataT;
  typedef typename Traits::LabelT LabelT;
  typedef typename Traits::IdxT IdxT;
  typedef typename Traits::NodeT NodeT;
  typedef typename Traits::SplitT SplitT;
  typedef typename Traits::InputT InputT;

  /** DT params */
  DecisionTreeParams params;
  /** input dataset */
  InputT input;

  /** max nodes that we can create */
  IdxT maxNodes;
  /** total number of histogram bins (classification only) */
  IdxT nHistBins;
  /** total number of prediction counts (regression only) */
  IdxT nPredCounts;
  /** size of block-sync workspace (regression + MAE only) */
  size_t block_sync_size;

  /** number of nodes created in the current batch */
  IdxT* n_nodes;
  /** class histograms (classification only) */
  int* hist;
  /** sum of predictions (regression only) */
  DataT* pred;
  /** MAE computation (regression only) */
  DataT* pred2;
  /** parent MAE computation (regression only) */
  DataT* pred2P;
  /** node count tracker for averaging (regression only) */
  IdxT* pred_count;
  /** threadblock arrival count */
  int* done_count;
  /** mutex array used for atomically updating best split */
  int* mutex;
  /** used for syncing across blocks in a kernel (regression + MAE only) */
  char* block_sync;
  /** number of leaves created so far */
  IdxT* n_leaves;
  /** max depth reached so far */
  IdxT* n_depth;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  NodeT* curr_nodes;
  /** next batch of nodes */
  NodeT* next_nodes;

  /** host copy of the number of new nodes in current branch */
  IdxT* h_n_nodes;
  /** total number of nodes created so far */
  IdxT h_total_nodes;
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
                     const DecisionTreeParams& p, const DataT* data,
                     const LabelT* labels, IdxT totalRows, IdxT totalCols,
                     IdxT sampledRows, IdxT sampledCols, IdxT* rowids,
                     IdxT* colids, IdxT nclasses, const DataT* quantiles) {
    ASSERT(quantiles != nullptr,
           "Currently quantiles need to be computed before this call!");
    params = p;
    params.n_blks_for_cols = std::min(sampledCols, p.n_blks_for_cols);
    input.data = data;
    input.labels = labels;
    input.M = totalRows;
    input.N = totalCols;
    input.nSampledRows = sampledRows;
    input.nSampledCols = sampledCols;
    input.rowids = rowids;
    input.colids = colids;
    input.nclasses = nclasses;
    input.quantiles = quantiles;
    auto max_batch = params.max_batch_size;
    auto n_col_blks = params.n_blks_for_cols;
    nHistBins = 2 * max_batch * params.n_bins * n_col_blks * nclasses;
    // x2 for mean and mean-of-square
    nPredCounts = max_batch * params.n_bins * n_col_blks;
    // x3 just to be safe since we can't strictly adhere to max_leaves
    maxNodes = params.max_leaves * 3;
    if (isRegression() && params.split_criterion == CRITERION::MAE) {
      dim3 grid(params.n_blks_for_rows, n_col_blks, max_batch);
      block_sync_size = MLCommon::GridSync::computeWorkspaceSize(
        grid, MLCommon::SyncType::ACROSS_X, false);
    } else {
      block_sync_size = 0;
    }
    d_wsize = 0;
    d_wsize += sizeof(IdxT);  // n_nodes
    if (!isRegression()) {
      d_wsize += sizeof(int) * nHistBins;  // hist
    } else {
      // x2 for left and right children
      d_wsize += 2 * nPredCounts * sizeof(DataT);  // pred
      d_wsize += 2 * nPredCounts * sizeof(DataT);  // pred2
      d_wsize += nPredCounts * sizeof(DataT);      // pred2P
      d_wsize += nPredCounts * sizeof(IdxT);       // pred_count
    }
    d_wsize += sizeof(int) * max_batch * n_col_blks;  // done_count
    d_wsize += sizeof(int) * max_batch;               // mutex
    d_wsize += block_sync_size;                       // block_sync
    d_wsize += sizeof(IdxT);                          // n_leaves
    d_wsize += sizeof(IdxT);                          // n_depth
    d_wsize += sizeof(SplitT) * max_batch;            // splits
    d_wsize += sizeof(NodeT) * max_batch;             // curr_nodes
    d_wsize += sizeof(NodeT) * 2 * max_batch;         // next_nodes
    // all nodes in the tree
    h_wsize = sizeof(IdxT);  // h_n_nodes
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
    if (!isRegression()) {
      hist = reinterpret_cast<int*>(d_wspace);
      d_wspace += sizeof(int) * nHistBins;
    } else {
      pred = reinterpret_cast<DataT*>(d_wspace);
      d_wspace += 2 * nPredCounts * sizeof(DataT);
      pred2 = reinterpret_cast<DataT*>(d_wspace);
      d_wspace += 2 * nPredCounts * sizeof(DataT);
      pred2P = reinterpret_cast<DataT*>(d_wspace);
      d_wspace += nPredCounts * sizeof(DataT);
      pred_count = reinterpret_cast<IdxT*>(d_wspace);
      d_wspace += nPredCounts * sizeof(IdxT);
    }
    done_count = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * max_batch * n_col_blks;
    mutex = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * max_batch;
    block_sync = reinterpret_cast<char*>(d_wspace);
    d_wspace += block_sync_size;
    n_leaves = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT);
    n_depth = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT);
    splits = reinterpret_cast<SplitT*>(d_wspace);
    d_wspace += sizeof(SplitT) * max_batch;
    curr_nodes = reinterpret_cast<NodeT*>(d_wspace);
    d_wspace += sizeof(NodeT) * max_batch;
    next_nodes = reinterpret_cast<NodeT*>(d_wspace);
    // host
    h_n_nodes = reinterpret_cast<IdxT*>(h_wspace);
  }

  /**
   * @brief Main training method. To be called only after `assignWorkspace()`
   * @param h_nodes list of nodes (must be allocated using cudaMallocHost!)
   * @param num_leaves number of leaves created in the tree
   * @param depth max depth of the built tree
   * @param s cuda steam
   */
  void train(NodeT* h_nodes, IdxT& num_leaves, IdxT& depth, cudaStream_t s) {
    init(h_nodes, s);
    while (true) {
      IdxT new_nodes = doSplit(h_nodes, s);
      h_total_nodes += new_nodes;
      if (new_nodes == 0 && isOver()) break;
      updateNodeRange();
    }
    MLCommon::updateHost(&num_leaves, n_leaves, 1, s);
    MLCommon::updateHost(&depth, n_depth, 1, s);
  }

 private:
  ///@todo: support starting from arbitrary nodes
  /**
   * @brief Initialize buffers and state
   * @param h_nodes list of nodes (must be allocated using cudaMallocHost!)
   * @param s cuda stream
   */
  void init(NodeT* h_nodes, cudaStream_t s) {
    *h_n_nodes = 0;
    auto max_batch = params.max_batch_size;
    auto n_col_blks = params.n_blks_for_cols;
    CUDA_CHECK(
      cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, s));
    CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, s));
    CUDA_CHECK(cudaMemsetAsync(n_leaves, 0, sizeof(IdxT), s));
    CUDA_CHECK(cudaMemsetAsync(n_depth, 0, sizeof(IdxT), s));
    node_start = 0;
    node_end = h_total_nodes = 1;  // start with root node
    h_nodes[0].initSpNode();
    h_nodes[0].start = 0;
    h_nodes[0].end = input.nSampledRows;
    h_nodes[0].depth = 0;
  }

  /** check whether any more nodes need to be processed or not */
  bool isOver() const { return node_end == h_total_nodes; }

  /**
   * @brief After the current batch is finished processing, update the range
   *        of nodes to be worked upon in the next batch
   */
  void updateNodeRange() {
    node_start = node_end;
    auto nodes_remaining = h_total_nodes - node_end;
    node_end = std::min(nodes_remaining, params.max_batch_size) + node_end;
  }

  /**
   * Computes best split across all nodes in the current batch and splits the
   * nodes accordingly
   * @param h_nodes list of nodes (must be allocated using cudaMallocHost!)
   * @param s cuda stream
   * @return the number of newly created nodes
   */
  IdxT doSplit(NodeT* h_nodes, cudaStream_t s) {
    auto batchSize = node_end - node_start;
    // start fresh on the number of *new* nodes created in this batch
    CUDA_CHECK(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), s));
    initSplit<DataT, IdxT, Traits::TPB_DEFAULT>(splits, batchSize, s);
    // get the current set of nodes to be worked upon
    MLCommon::updateDevice(curr_nodes, h_nodes + node_start, batchSize, s);
    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    auto n_col_blks = params.n_blks_for_cols;
    for (IdxT c = 0; c < input.nSampledCols; c += n_col_blks) {
      Traits::computeSplit(*this, c, batchSize, params.split_criterion, s);
      CUDA_CHECK(cudaGetLastError());
    }
    // create child nodes (or make the current ones leaf)
    Traits::nodeSplit(*this, batchSize, s);
    CUDA_CHECK(cudaGetLastError());
    // copy the updated (due to leaf creation) and newly created child nodes
    MLCommon::updateHost(h_nodes + node_start, curr_nodes, batchSize, s);
    MLCommon::updateHost(h_n_nodes, n_nodes, 1, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    MLCommon::updateHost(h_nodes + h_total_nodes, next_nodes, *h_n_nodes, s);
    return *h_n_nodes;
  }
};  // end Builder

/**
 * @brief Traits used to customize the Builder for classification task
 * @tparam _data data type
 * @tparam _label label type
 * @tparam _idx index type
 */
template <typename _data, typename _label, typename _idx>
struct ClsTraits {
  typedef _data DataT;
  typedef _label LabelT;
  typedef _idx IdxT;
  typedef Node<DataT, LabelT, IdxT> NodeT;
  typedef Split<DataT, IdxT> SplitT;
  typedef Input<DataT, LabelT, IdxT> InputT;

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 256;
  /** threads per block for the nodeSplitKernel */
  static constexpr int TPB_SPLIT = 512;

  /**
   * @brief Compute best split for the currently given set of columns
   * @param b builder object
   * @param col start column id
   * @param batchSize number of nodes to be processed in this call
   * @param splitType split criterion
   * @param s cuda stream
   */
  static void computeSplit(Builder<ClsTraits<DataT, LabelT, IdxT>>& b, IdxT col,
                           IdxT batchSize, CRITERION splitType,
                           cudaStream_t s) {
    auto nbins = b.params.n_bins;
    auto nclasses = b.input.nclasses;
    auto binSize = nbins * 2 * nclasses;
    auto colBlks =
      std::min(b.params.n_blks_for_cols, b.input.nSampledCols - col);
    dim3 grid(b.params.n_blks_for_rows, colBlks, batchSize);
    size_t smemSize = sizeof(int) * binSize + sizeof(DataT) * nbins;
    smemSize += sizeof(int);
    CUDA_CHECK(cudaMemsetAsync(b.hist, 0, sizeof(int) * b.nHistBins, s));
    computeSplitClassificationKernel<DataT, LabelT, IdxT, TPB_DEFAULT>
      <<<grid, TPB_DEFAULT, smemSize, s>>>(
        b.hist, b.params.n_bins, b.params.max_depth, b.params.min_rows_per_node,
        b.params.max_leaves, b.input, b.curr_nodes, col, b.done_count, b.mutex,
        b.n_leaves, b.splits, splitType);
  }

  /**
   * @brief Split the node into left/right children
   * @param b builder object
   * @param batchSize number of nodes to be processed in this call
   * @param s cuda stream
   */
  static void nodeSplit(Builder<ClsTraits<DataT, LabelT, IdxT>>& b,
                        IdxT batchSize, cudaStream_t s) {
    auto smemSize =
      std::max(2 * sizeof(IdxT) * TPB_SPLIT, sizeof(int) * b.input.nclasses);
    nodeSplitClassificationKernel<DataT, LabelT, IdxT, TPB_SPLIT>
      <<<batchSize, TPB_SPLIT, smemSize, s>>>(
        b.params.max_depth, b.params.min_rows_per_node, b.params.max_leaves,
        b.params.min_impurity_decrease, b.input, b.curr_nodes, b.next_nodes,
        b.n_nodes, b.splits, b.n_leaves, b.h_total_nodes, b.n_depth);
  }
};  // end ClsTraits

/**
 * @brief Traits used to customize the Builder for regression task
 * @tparam _data data type
 * @tparam _idx index type
 * @note label type is assumed to be the same as input data type
 */
template <typename _data, typename _idx>
struct RegTraits {
  typedef _data DataT;
  typedef _data LabelT;
  typedef _idx IdxT;
  typedef Node<DataT, LabelT, IdxT> NodeT;
  typedef Split<DataT, IdxT> SplitT;
  typedef Input<DataT, LabelT, IdxT> InputT;

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 256;
  /** threads per block for the nodeSplitKernel */
  static constexpr int TPB_SPLIT = 512;

  /**
   * @brief Compute best split for the currently given set of columns
   * @param b builder object
   * @param col start column id
   * @param batchSize number of nodes to be processed in this call
   * @param splitType split criterion
   * @param s cuda stream
   */
  static void computeSplit(Builder<RegTraits<DataT, IdxT>>& b, IdxT col,
                           IdxT batchSize, CRITERION splitType,
                           cudaStream_t s) {
    auto n_col_blks = b.params.n_blks_for_cols;
    dim3 grid(b.params.n_blks_for_rows, n_col_blks, batchSize);
    auto nbins = b.params.n_bins;
    size_t smemSize = 7 * nbins * sizeof(DataT) + nbins * sizeof(int);
    smemSize += sizeof(int);
    CUDA_CHECK(
      cudaMemsetAsync(b.pred, 0, sizeof(DataT) * b.nPredCounts * 2, s));
    if (splitType == CRITERION::MAE) {
      CUDA_CHECK(
        cudaMemsetAsync(b.pred2, 0, sizeof(DataT) * b.nPredCounts * 2, s));
      CUDA_CHECK(
        cudaMemsetAsync(b.pred2P, 0, sizeof(DataT) * b.nPredCounts, s));
    }
    CUDA_CHECK(
      cudaMemsetAsync(b.pred_count, 0, sizeof(IdxT) * b.nPredCounts, s));
    computeSplitRegressionKernel<DataT, DataT, IdxT, TPB_DEFAULT>
      <<<grid, TPB_DEFAULT, smemSize, s>>>(
        b.pred, b.pred2, b.pred2P, b.pred_count, b.params.n_bins,
        b.params.max_depth, b.params.min_rows_per_node, b.params.max_leaves,
        b.input, b.curr_nodes, col, b.done_count, b.mutex, b.n_leaves, b.splits,
        b.block_sync, splitType);
  }

  /**
   * @brief Split the node into left/right children
   * @param b builder object
   * @param batchSize number of nodes to be processed in this call
   * @param s cuda stream
   */
  static void nodeSplit(Builder<RegTraits<DataT, IdxT>>& b, IdxT batchSize,
                        cudaStream_t s) {
    auto smemSize = 2 * sizeof(IdxT) * TPB_SPLIT;
    nodeSplitRegressionKernel<DataT, IdxT, TPB_SPLIT>
      <<<batchSize, TPB_SPLIT, smemSize, s>>>(
        b.params.max_depth, b.params.min_rows_per_node, b.params.max_leaves,
        b.params.min_impurity_decrease, b.input, b.curr_nodes, b.next_nodes,
        b.n_nodes, b.splits, b.n_leaves, b.h_total_nodes, b.n_depth);
  }
};  // end RegTraits

}  // namespace DecisionTree
}  // namespace ML
