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

namespace ML {
namespace decisiontree {

/**
 * Internal struct used to do all the heavy-lifting required for tree building
 *
 * @note This struct does NOT own any of the underlying device/host pointers.
 *       They all must explicitly be allocated by the caller and passed to it.
 */
template <typename DataT, typename LabelT, typename IdxT>
struct Builder {
  typedef Node<DataT, LabelT, IdxT> NodeT;
  typedef SparseTree<DataT, LabelT, IdxT> SpTreeT;
  typedef Split<DataT, IdxT> SplitT;

  /** number of sampled rows */
  IdxT nrows;
  /** number of sampled columns */
  IdxT ncols;
  /** max nodes that we can create */
  IdxT max_nodes;
  /** number of blocks used to parallelize column-wise computations */
  IdxT nBlksForCols;
  /** total number of histogram bins */
  IdxT nHistBins;
  /** DT params */
  Params<DataT, IdxT> params;
  /** training input */
  Input<DataT, LabelT, IdxT> input;
  /** will contain the final learned tree */
  SpTreeT tree;
  /** gain before splitting root node */
  DataT rootGain;

  /** sampled row id's */
  IdxT* rowids;
  /** sampled column id's */
  ///@todo: support for per-node col-subsampling
  IdxT* colids;
  /** number of nodes created in the current batch */
  IdxT* n_nodes;
  /** quantiles computed on the dataset (col-major) */
  DataT* quantiles;
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
  IdxT* h_n_nodes;
  /** host copy for initial histograms */
  int* h_hist;

  /** checks if this struct is being used for classification or regression */
  static constexpr bool isRegression() {
    return std::is_same<DataT, LabelT>::value;
  }

  /**
   * @brief Computes workspace size needed for the current computation
   * @param d_wsize (in B) of the device workspace to be allocated
   * @param h_wsize (in B) of the host workspace to be allocated
   * @param p the input params
   * @param in the input data
   */
  void workspaceSize(size_t& d_wsize, size_t& h_wsize,
                     const Params<DataT, IdxT>& p,
                     const Input<DataT, LabelT, IdxT>& in) {
    ASSERT(!isRegression(), "Currently only classification is supported!");
    nrows = static_cast<IdxT>(p.row_subsample * in.M);
    ncols = static_cast<IdxT>(p.col_subsample * in.N);
    nBlksForCols = std::min(ncols, p.nBlksForCols);
    auto max_batch = params.max_batch_size;
    nHistBins = 2 * max_batch * (p.nbins + 1) * nBlksForCols;
    // x3 just to be safe since we can't strictly adhere to max_leaves
    max_nodes = p.max_leaves * 3;
    params = p;
    input = in;
    d_wsize = static_cast<size_t>(0);
    d_wsize += sizeof(IdxT) * nrows;                    // rowids
    d_wsize += sizeof(IdxT) * ncols;                    // colids
    d_wsize += sizeof(IdxT);                            // n_nodes
    d_wsize += sizeof(DataT) * p.nbins * in.N;          // quantiles
    d_wsize += sizeof(int) * nHistBins;                 // hist
    d_wsize += sizeof(int) * max_batch * nBlksForCols;  // done_count
    d_wsize += sizeof(int) * max_batch;                 // mutex
    d_wsize += sizeof(IdxT);                            // n_leaves
    d_wsize += sizeof(SplitT) * max_batch;              // splits
    d_wsize += sizeof(NodeT) * max_batch;               // curr_nodes
    d_wsize += sizeof(NodeT) * 2 * max_batch;           // next_nodes
    // all nodes in the tree
    h_wsize = sizeof(IdxT);                   // h_n_nodes
    h_wsize += sizeof(int) * input.nclasses;  // h_hist
    h_wsize += (sizeof(NodeT) + sizeof(SplitT)) * max_nodes;
  }

  /**
   * @brief assign workspace to the current state
   * @param d_wspace device buffer allocated by the user for the workspace. Its
   *                 size should be atleast workspaceSize()
   * @param h_wspace pinned host buffer mainly needed to store the learned nodes
   * @param s cuda stream where to schedule work
   */
  void assignWorkspace(char* d_wspace, char* h_wspace) {
    auto max_batch = params.max_batch_size;
    // device
    rowids = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT) * nrows;
    colids = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT) * ncols;
    n_nodes = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT);
    quantiles = reinterpret_cast<DataT*>(d_wspace);
    d_wspace += sizeof(DataT) * params.nbins * ncols;
    hist = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * nHistBins;
    done_count = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * max_batch * nBlksForCols;
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
    h_hist = reinterpret_cast<int*>(h_wspace);
    h_wspace += sizeof(IdxT) * input.nclasses;
    tree.assignWorkspace(h_wspace, max_nodes);
  }

  /**
   * @brief Main training method. To be called only after `assignWorkspace()`
   * @param bootstrap whether to bootstrap the rows or sample w/o replacement
   * @param allocator device allocator
   * @param s cuda stream
   * @param seed seed to initialize the underlying RNG (for reproducibility)
   */
  void subsampleRows(bool bootstrap,
                     const std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t s, uint64_t seed = 0ULL) {
    if (bootstrap) {
      if (seed == 0ULL) seed = static_cast<uint64_t>(time(NULL));
      MLCommon::Random::Rng r(seed);
      r.uniformInt(rowids, nrows, 0, nrows, s);
    } else {
      MLCommon::device_buffer<IdxT> outkeys(allocator, s, input.M);
      IdxT* outPtr = outkeys.data();
      int* dummy = nullptr;
      MLCommon::Random::permute(outPtr, dummy, dummy, 1, input.M, false, s);
      // outkeys has more rows than selected_rows; doing the shuffling before
      // the resize to differentiate the per-tree rows sample.
      CUDA_CHECK(cudaMemcpyAsync(rowids, outPtr, nrows * sizeof(IdxT),
                                 cudaMemcpyDeviceToDevice, s));
      outkeys.release(s);
    }
  }

  /** Main training method. To be called only after `subsampleRows()` */
  void train(cudaStream_t s) {
    init(s);
    while (!tree.isOver()) {
      auto new_nodes = doSplit(s);
      tree.n_nodes += new_nodes;
      tree.updateNodeRange(params.max_batch_size);
    }
  }

 private:
  void init(cudaStream_t s) {
    auto max_batch = params.max_batch_size;
    CUDA_CHECK(cudaMemsetAsync(done_count, 0,
                               sizeof(int) * max_batch * nBlksForCols, s));
    CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, s));
    CUDA_CHECK(cudaMemsetAsync(n_leaves, 0, sizeof(IdxT), s));
    rootGain = initialMetric(s);
    tree.init(nrows, rootGain);
  }

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 256;
  /** threads per block for the nodeSplitKernel */
  static constexpr int TPB_SPLIT = 512;

  /**
   * Computes best split across all nodes in the current batch and splits the
   * nodes accordingly
   * @return the number of newly created nodes
   */
  IdxT doSplit(cudaStream_t s) {
    auto nbins = params.nbins;
    auto nclasses = input.nclasses;
    auto binSize = nbins * 2 * nclasses;
    auto len = binSize + 2 * nbins;
    size_t smemSize = sizeof(int) * len + sizeof(DataT) * nbins;
    auto batchSize = tree.end - tree.start;
    auto nblks = MLCommon::ceildiv<int>(batchSize, TPB_DEFAULT);
    // start fresh on the number of *new* nodes created in this batch
    CUDA_CHECK(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), s));
    initSplitKernel<DataT, IdxT>
      <<<nblks, TPB_DEFAULT, 0, s>>>(splits, batchSize);
    CUDA_CHECK(cudaGetLastError());
    // get the current set of nodes to be worked upon
    MLCommon::updateDevice(curr_nodes, tree.nodes + tree.start, batchSize, s);
    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    dim3 grid(params.nBlksForRows, nBlksForCols, batchSize);
    for (IdxT c = 0; c < ncols; c += nBlksForCols) {
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
                                              splits, n_leaves, tree.n_nodes);
    CUDA_CHECK(cudaGetLastError());
    // copy the best splits to host
    MLCommon::updateHost(tree.splits + tree.start, splits, batchSize, s);
    // copy the updated (due to leaf creation) and newly created child nodes
    MLCommon::updateHost(tree.nodes + tree.start, curr_nodes, batchSize, s);
    MLCommon::updateHost(h_n_nodes, n_nodes, 1, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    MLCommon::updateHost(tree.nodes + tree.n_nodes, next_nodes, *h_n_nodes, s);
    return *h_n_nodes;
  }

  /** computes the initial metric needed for root node split decision */
  DataT initialMetric(cudaStream_t s) {
    static constexpr int TPB = 256;
    static constexpr int NITEMS = 8;
    int nblks = ceildiv(nrows, TPB * NITEMS);
    size_t smemSize = sizeof(int) * input.nclasses;
    auto out = DataT(1.0);
    ///@todo: support for regression
    if (isRegression()) {
    } else {
      // reusing `hist` for initial bin computation only
      CUDA_CHECK(cudaMemsetAsync(hist, 0, sizeof(int) * input.nclasses, s));
      initialClassHistKernel<DataT, LabelT, IdxT><<<nblks, TPB, smemSize, s>>>(
        hist, rowids, input.labels, input.nclasses, nrows);
      CUDA_CHECK(cudaGetLastError());
      MLCommon::updateHost(h_hist, hist, input.nclasses, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      // better to compute the initial metric (after class histograms) on CPU
      ///@todo: support other metrics
      auto invlen = out / DataT(nrows);
      for (IdxT i = 0; i < input.nclasses; ++i) {
        auto val = h_hist[i] * invlen;
        out -= val * val;
      }
    }
    return out;
  }
};  // end Builder

}  // namespace decisiontree
}  // namespace ML
