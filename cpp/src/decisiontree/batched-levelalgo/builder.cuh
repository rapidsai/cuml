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
  /** total number of histogram bins (classification only) */
  IdxT nHistBins;
  /** total number of prediction counts (regression only) */
  IdxT nPredCounts;
  /** total number of prediction summations (regression only) */
  IdxT nPreds;
  /** gain/metric before splitting root node */
  DataT rootGain;

  /** number of nodes created in the current batch */
  IdxT* n_nodes;
  /** class histograms (classification only) */
  int* hist;
  /** sum of predictions (regression only) */
  DataT* pred;
  /** sum of squared of predictions (regression only) */
  DataT* pred2;
  /** node count tracker for averaging (regression only) */
  IdxT* pred_count;
  /** threadblock arrival count */
  int* done_count;
  /** mutex array used for atomically updating best split */
  int* mutex;
  /** number of leaves created so far */
  IdxT* n_leaves;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  NodeT* curr_nodes;
  /** next batch of nodes */
  NodeT* next_nodes;

  /** host copy of the number of new nodes in current branch */
  IdxT* h_n_nodes;
  /** host copy for initial histograms (classification only) */
  int* h_hist;
  /** host copy for MSE computation (regression only) */
  DataT* h_mse;
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
    ASSERT(!isRegression(), "Currently only classification is supported!");
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
    nPreds = 2 * nPredCounts;
    // x3 just to be safe since we can't strictly adhere to max_leaves
    maxNodes = params.max_leaves * 3;
    d_wsize = 0;
    d_wsize += sizeof(IdxT);  // n_nodes
    if (!isRegression()) {
      d_wsize += sizeof(int) * nHistBins;  // hist
    } else {
      // x2 for left and right children
      d_wsize += 2 * nPreds * sizeof(DataT);  // pred
      d_wsize += 2 * nPreds * sizeof(DataT);  // pred2
      d_wsize += nPredCounts * sizeof(IdxT);  // pred_count
    }
    d_wsize += sizeof(int) * max_batch * n_col_blks;  // done_count
    d_wsize += sizeof(int) * max_batch;               // mutex
    d_wsize += sizeof(IdxT);                          // n_leaves
    d_wsize += sizeof(SplitT) * max_batch;            // splits
    d_wsize += sizeof(NodeT) * max_batch;             // curr_nodes
    d_wsize += sizeof(NodeT) * 2 * max_batch;         // next_nodes
    // all nodes in the tree
    h_wsize = sizeof(IdxT);  // h_n_nodes
    if (!isRegression()) {
      h_wsize += sizeof(int) * nclasses;  // h_hist
    } else {
      h_wsize += sizeof(DataT) * 2;  // h_mse
    }
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
      d_wspace += 2 * nPreds * sizeof(DataT);
      pred2 = reinterpret_cast<DataT*>(d_wspace);
      d_wspace += 2 * nPreds * sizeof(DataT);
      pred_count = reinterpret_cast<IdxT*>(d_wspace);
      d_wspace += nPredCounts * sizeof(IdxT);
    }
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
    h_n_nodes = reinterpret_cast<IdxT*>(h_wspace);
    h_wspace += sizeof(IdxT);
    if (!isRegression()) {
      h_hist = reinterpret_cast<int*>(h_wspace);
    } else {
      h_mse = reinterpret_cast<DataT*>(h_wspace);
    }
  }

  /**
   * @brief Main training method. To be called only after `assignWorkspace()`
   * @param h_nodes list of nodes (must be allocated using cudaMallocHost!)
   * @param s cuda steam
   */
  void train(NodeT* h_nodes, cudaStream_t s) {
    init(h_nodes, s);
    do {
      IdxT new_nodes;
      if (params.split_criterion == CRITERION::GINI) {
        new_nodes = doSplit<CRITERION::GINI>(h_nodes, s);
      } else {
        new_nodes = doSplit<CRITERION::ENTROPY>(h_nodes, s);
      }
      h_total_nodes += new_nodes;
      updateNodeRange();
    } while (!isOver());
  }

 private:
  ///@todo: support starting from arbitrary nodes
  /**
   * @brief Initialize buffers and state
   * @param h_nodes list of nodes (must be allocated using cudaMallocHost!)
   * @param s cuda stream
   */
  void init(NodeT* h_nodes, cudaStream_t s) {
    auto max_batch = params.max_batch_size;
    auto n_col_blks = params.n_blks_for_cols;
    CUDA_CHECK(
      cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, s));
    CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, s));
    CUDA_CHECK(cudaMemsetAsync(n_leaves, 0, sizeof(IdxT), s));
    rootGain = initialMetric(s);
    node_start = 0;
    node_end = h_total_nodes = 1;  // start with root node
    h_nodes[0].parentGain = rootGain;
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

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 256;
  /** threads per block for the nodeSplitKernel */
  static constexpr int TPB_SPLIT = 512;

  /**
   * Computes best split across all nodes in the current batch and splits the
   * nodes accordingly
   * @param h_nodes list of nodes (must be allocated using cudaMallocHost!)
   * @param s cuda stream
   * @return the number of newly created nodes
   */
  template <CRITERION SplitType>
  IdxT doSplit(NodeT* h_nodes, cudaStream_t s) {
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
    for (IdxT c = 0; c < input.nSampledCols; c += n_col_blks) {
      CUDA_CHECK(cudaMemsetAsync(hist, 0, sizeof(int) * nHistBins, s));
      computeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT, SplitType>
        <<<grid, TPB_DEFAULT, smemSize, s>>>(
          hist, params.n_bins, params.max_depth, params.min_rows_per_node,
          params.max_leaves, input, curr_nodes, c, done_count, mutex, n_leaves,
          splits);
      CUDA_CHECK(cudaGetLastError());
    }
    // create child nodes (or make the current ones leaf)
    smemSize = std::max(2 * sizeof(IdxT) * TPB_SPLIT, sizeof(int) * nclasses);
    nodeSplitKernel<DataT, LabelT, IdxT, TPB_SPLIT>
      <<<batchSize, TPB_SPLIT, smemSize, s>>>(
        params.max_depth, params.min_rows_per_node, params.max_leaves,
        params.min_impurity_decrease, input, curr_nodes, next_nodes, n_nodes,
        splits, n_leaves, h_total_nodes);
    CUDA_CHECK(cudaGetLastError());
    // copy the updated (due to leaf creation) and newly created child nodes
    MLCommon::updateHost(h_nodes + node_start, curr_nodes, batchSize, s);
    MLCommon::updateHost(h_n_nodes, n_nodes, 1, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    MLCommon::updateHost(h_nodes + h_total_nodes, next_nodes, *h_n_nodes, s);
    return *h_n_nodes;
  }

  /** computes the initial metric needed for root node split decision */
  DataT initialMetric(cudaStream_t s) {
    static constexpr int NITEMS = 8;
    auto nblks =
      MLCommon::ceildiv<int>(input.nSampledRows, TPB_DEFAULT * NITEMS);
    size_t smemSize = sizeof(int) * input.nclasses;
    auto out = DataT(0.0);
    if (isRegression()) {
      // reusing `pred` for initial mse computation only
      CUDA_CHECK(cudaMemsetAsync(pred, 0, sizeof(DataT) * 2, s));
      initialMeanPredKernel<DataT, LabelT, IdxT><<<nblks, TPB_DEFAULT, 0, s>>>(
        pred, pred + 1, input.rowids, input.labels, input.nSampledRows);
      CUDA_CHECK(cudaGetLastError());
      MLCommon::updateHost(h_mse, pred, 2, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      ///@todo: add support for other regression metrics (currently MSE only)
      out = h_mse[1] - h_mse[0] * h_mse[0];
    } else {
      // reusing `hist` for initial bin computation only
      CUDA_CHECK(cudaMemsetAsync(hist, 0, sizeof(int) * input.nclasses, s));
      initialClassHistKernel<DataT, LabelT, IdxT>
        <<<nblks, TPB_DEFAULT, smemSize, s>>>(
          hist, input.rowids, input.labels, input.nclasses, input.nSampledRows);
      CUDA_CHECK(cudaGetLastError());
      MLCommon::updateHost(h_hist, hist, input.nclasses, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      // better to compute the initial metric (after class histograms) on CPU
      if (params.split_criterion == CRITERION::GINI) {
        out =
          giniMetric<DataT, IdxT>(h_hist, input.nclasses, input.nSampledRows);
      } else {
        out = entropyMetric<DataT, IdxT>(h_hist, input.nclasses,
                                         input.nSampledRows);
      }
    }
    return out;
  }
};  // end Builder

///@todo: support regression
///@todo: support building from an arbitrary depth
///@todo: support col subsampling per node
/**
 * @brief Main entry point function for batched-level algo to build trees
 * @param d_allocator device allocator
 * @param h_allocator host allocator
 * @param data input dataset (on device) (col-major) (dim = nrows x ncols)
 * @param ncols number of features in the dataset
 * @param nrows number of rows in the dataset
 * @param labels labels for the input dataset (on device) (len = nrows)
 * @param quantiles histograms/quantiles of the input dataset (on device)
 *                  (col-major) (dim = params.n_bins x ncols)
 * @param rowids sampled rows (on device) (len = n_sampled_rows)
 * @param colids sampled cols (on device) (len = params.max_features * ncols)
 * @param n_sampled_rows number of sub-sampled rows
 * @param unique_labels number of classes (meaningful only for classification)
 * @param params decisiontree learning params
 * @param stream cuda stream
 * @param sparsetree output learned tree
 */
template <typename DataT, typename LabelT, typename IdxT>
void grow_tree(std::shared_ptr<MLCommon::deviceAllocator> d_allocator,
               std::shared_ptr<MLCommon::hostAllocator> h_allocator,
               const DataT* data, IdxT ncols, IdxT nrows, const LabelT* labels,
               const DataT* quantiles, IdxT* rowids, IdxT* colids,
               int n_sampled_rows, int unique_labels,
               const DecisionTreeParams& params, cudaStream_t stream,
               std::vector<SparseTreeNode<DataT, LabelT>>& sparsetree) {
  Builder<DataT, LabelT, IdxT> builder;
  size_t d_wsize, h_wsize;
  builder.workspaceSize(d_wsize, h_wsize, params, data, labels, nrows, ncols,
                        n_sampled_rows, IdxT(params.max_features * ncols),
                        rowids, colids, unique_labels, quantiles);
  MLCommon::device_buffer<char> d_buff(d_allocator, stream, d_wsize);
  MLCommon::host_buffer<char> h_buff(h_allocator, stream, h_wsize);
  MLCommon::host_buffer<Node<DataT, LabelT, IdxT>> h_nodes(h_allocator, stream,
                                                           builder.maxNodes);
  builder.assignWorkspace(d_buff.data(), h_buff.data());
  builder.train(h_nodes.data(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  d_buff.release(stream);
  h_buff.release(stream);
  ///@todo: copy from Node to sparsetree
  h_nodes.release(stream);
}

}  // namespace DecisionTree
}  // namespace ML
