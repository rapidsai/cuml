/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <common/grid_sync.cuh>
#include <cuml/common/logger.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <raft/cuda_utils.cuh>
#include "input.cuh"
#include "kernels.cuh"
#include "metrics.cuh"
#include "node.cuh"
#include "split.cuh"

#include <common/nvtx.hpp>
#include <deque>
#include <utility>

namespace ML {
namespace DT {

template <typename NodeT>
class NodeQueue {
  const DecisionTreeParams params;
  std::vector<NodeT> tree_;
  std::deque<NodeWorkItem> work_items_;
  int tree_depth_ = 0;
  int num_leaves_ = 1;

 public:
  NodeQueue(DecisionTreeParams params, size_t max_nodes, size_t sampled_rows) : params(params)
  {
    tree_.reserve(max_nodes);
    tree_.emplace_back(0, sampled_rows, 0);
    if (this->IsExpandable(tree_.back())) {
      work_items_.emplace_back(NodeWorkItem{0, 0, sampled_rows, 0});
    }
  }

  int TreeDepth() { return tree_depth_; }
  int NumLeaves() { return num_leaves_; }
  std::vector<NodeT> GetTree() { return tree_; }

  bool HasWork() { return work_items_.size() > 0; }

  auto Pop()
  {
    std::vector<NodeWorkItem> result;
    result.reserve(std::min(size_t(params.max_batch_size), work_items_.size()));
    while (work_items_.size() > 0 && result.size() < std::size_t(params.max_batch_size)) {
      result.emplace_back(work_items_.front());
      work_items_.pop_front();
    }
    return result;
  }

  // This node is allowed to be expanded further (if its split gain is high enough)
  bool IsExpandable(const NodeT& n)
  {
    if (n.depth >= params.max_depth) return false;
    if (n.count < params.min_samples_split) return false;
    if (params.max_leaves != -1 && num_leaves_ >= params.max_leaves) return false;
    return true;
  }

  template <typename SplitT>
  void Push(const std::vector<NodeWorkItem>& work_items, SplitT* h_splits)
  {
    // Update node queue based on splits
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto split      = h_splits[i];
      auto item       = work_items[i];
      const auto node = tree_[item.idx];
      if (SplitNotValid(split, params.min_impurity_decrease, params.min_samples_leaf, node.count)) {
        continue;
      }

      if (params.max_leaves != -1 && num_leaves_ >= params.max_leaves) break;

      // parent
      tree_[item.idx] = NodeT::CreateSplit(
        split.colid, split.quesval, split.best_metric_val, tree_.size(), node.count);
      num_leaves_++;
      // left
      tree_.emplace_back(NodeT::CreateChild(node.depth + 1, node.start, split.nLeft));

      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree_.back())) {
        work_items_.emplace_back(
          NodeWorkItem{tree_.size() - 1, size_t(node.start), size_t(split.nLeft), node.depth + 1});
      }

      // right
      tree_.emplace_back(
        NodeT::CreateChild(node.depth + 1, node.start + split.nLeft, node.count - split.nLeft));
      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree_.back())) {
        work_items_.emplace_back(NodeWorkItem{tree_.size() - 1,
                                              size_t(node.start + split.nLeft),
                                              size_t(node.count - split.nLeft),
                                              node.depth + 1});
      }

      // update depth
      tree_depth_ = max(tree_depth_, node.depth + 1);
    }
  }
};

/**
 * Internal struct used to do all the heavy-lifting required for tree building
 *
 * @note This struct does NOT own any of the underlying device/host pointers.
 *       They all must explicitly be allocated by the caller and passed to it.
 */
template <typename ObjectiveT>
struct Builder {
  typedef typename ObjectiveT::DataT DataT;
  typedef typename ObjectiveT::LabelT LabelT;
  typedef typename ObjectiveT::IdxT IdxT;
  typedef typename ObjectiveT::BinT BinT;
  typedef Node<DataT, LabelT, IdxT> NodeT;
  typedef Split<DataT, IdxT> SplitT;
  typedef Input<DataT, LabelT, IdxT> InputT;

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 128;
  const raft::handle_t& handle;
  /** DT params */
  DecisionTreeParams params;
  /** input dataset */
  InputT input;

  /** Tree index */
  IdxT treeid;
  /** Seed used for randomization */
  uint64_t seed;
  /** number of nodes created in the current batch */
  IdxT* n_nodes;
  /** histograms */
  BinT* hist;
  /** threadblock arrival count */
  int* done_count;
  /** mutex array used for atomically updating best split */
  int* mutex;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  NodeWorkItem* d_work_items;

  WorkloadInfo<IdxT>* workload_info;
  WorkloadInfo<IdxT>* h_workload_info;

  int max_blocks = 0;

  SplitT* h_splits;
  /** number of blocks used to parallelize column-wise computations. */
  int n_blks_for_cols = 10;
  /** Memory alignment value */
  const size_t alignValue = 512;

  rmm::device_uvector<char> d_buff;
  std::vector<char> h_buff;

  Builder(const raft::handle_t& handle,
          IdxT treeid,
          uint64_t seed,
          const DecisionTreeParams& p,
          const DataT* data,
          const LabelT* labels,
          IdxT totalRows,
          IdxT totalCols,
          IdxT sampledRows,
          IdxT* rowids,
          IdxT nclasses,
          const DataT* quantiles)
    : handle(handle),
      treeid(treeid),
      seed(seed),
      params(p),
      input{data,
            labels,
            totalRows,
            totalCols,
            sampledRows,
            max(1, IdxT(params.max_features * totalCols)),
            rowids,
            nclasses,
            quantiles},
      d_buff(0, handle.get_stream()),
      h_buff(0)
  {
    max_blocks = 1 + params.max_batch_size + input.nSampledRows / TPB_DEFAULT;
    ASSERT(quantiles != nullptr, "Currently quantiles need to be computed before this call!");
    ASSERT(nclasses >= 1, "nclasses should be at least 1");

    auto [device_workspace_size, host_workspace_size] = workspaceSize();
    d_buff.resize(device_workspace_size, handle.get_stream());
    h_buff.resize(host_workspace_size);
    assignWorkspace(d_buff.data(), h_buff.data());
  }

  size_t calculateAlignedBytes(const size_t actualSize) const
  {
    return raft::alignTo(actualSize, alignValue);
  }

  size_t maxNodes() const
  {
    if (params.max_depth < 13) {
      // Start with allocation for a dense tree for depth < 13
      return pow(2, (params.max_depth + 1)) - 1;
    } else {
      // Start with fixed size allocation for depth >= 13
      return 8191;
    }
  }

  auto workspaceSize() const
  {
    size_t d_wsize = 0, h_wsize = 0;
    ML::PUSH_RANGE("Builder::workspaceSize @builder_base.cuh [batched-levelalgo]");
    auto max_batch   = params.max_batch_size;
    size_t nHistBins = max_batch * (params.n_bins) * n_blks_for_cols * input.numOutputs;

    d_wsize += calculateAlignedBytes(sizeof(IdxT));                               // n_nodes
    d_wsize += calculateAlignedBytes(sizeof(BinT) * nHistBins);                   // hist
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch * n_blks_for_cols);  // done_count
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch);                    // mutex
    d_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);                 // splits
    d_wsize += calculateAlignedBytes(sizeof(NodeWorkItem) * max_batch);           // d_work_Items
    d_wsize +=                                                                    // workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);

    // all nodes in the tree
    h_wsize +=  // h_workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);
    h_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);  // splits

    ML::POP_RANGE();
    return std::make_pair(d_wsize, h_wsize);
  }

  /**
   * @brief assign workspace to the current state
   *
   * @param[in] d_wspace device buffer allocated by the user for the workspace.
   *                     Its size should be atleast workspaceSize()
   * @param[in] h_wspace pinned host buffer needed to store the learned nodes
   */
  void assignWorkspace(char* d_wspace, char* h_wspace)
  {
    ML::PUSH_RANGE("Builder::assignWorkspace @builder_base.cuh [batched-levelalgo]");
    auto max_batch   = params.max_batch_size;
    auto n_col_blks  = n_blks_for_cols;
    size_t nHistBins = max_batch * (params.n_bins) * n_blks_for_cols * input.numOutputs;
    // device
    n_nodes = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT));
    hist = reinterpret_cast<BinT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(BinT) * nHistBins);
    done_count = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(int) * max_batch * n_col_blks);
    mutex = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(int) * max_batch);
    splits = reinterpret_cast<SplitT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    d_work_items = reinterpret_cast<NodeWorkItem*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(NodeWorkItem) * max_batch);
    workload_info = reinterpret_cast<WorkloadInfo<IdxT>*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);

    CUDA_CHECK(
      cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, handle.get_stream()));
    CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, handle.get_stream()));

    // host
    h_workload_info = reinterpret_cast<WorkloadInfo<IdxT>*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);
    h_splits = reinterpret_cast<SplitT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    ML::POP_RANGE();
  }

  std::vector<Node<DataT, LabelT, IdxT>> train(IdxT& num_leaves, IdxT& depth, cudaStream_t s)
  {
    ML::PUSH_RANGE("Builder::train @builder_base.cuh [batched-levelalgo]");
    NodeQueue<NodeT> queue(params, this->maxNodes(), input.nSampledRows);
    while (queue.HasWork()) {
      auto work_items                      = queue.Pop();
      auto [splits_host_ptr, splits_count] = doSplit(work_items);
      queue.Push(work_items, splits_host_ptr);
    }
    depth      = queue.TreeDepth();
    num_leaves = queue.NumLeaves();
    auto tree  = queue.GetTree();
    this->SetLeafPredictions(&tree);
    ML::POP_RANGE();
    return tree;
  }

 private:
  auto updateWorkloadInfo(const std::vector<NodeWorkItem>& work_items)
  {
    int n_large_nodes_in_curr_batch =
      0;  // large nodes are nodes having training instances larger than block size, hence require
          // global memory for histogram construction
    int total_num_blocks = 0;
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto item      = work_items[i];
      int num_blocks = raft::ceildiv(item.row_count, size_t(TPB_DEFAULT));
      num_blocks     = std::max(1, num_blocks);

      if (num_blocks > 1) ++n_large_nodes_in_curr_batch;

      for (int b = 0; b < num_blocks; b++) {
        h_workload_info[total_num_blocks + b] = {
          int(i), n_large_nodes_in_curr_batch - 1, b, num_blocks};
      }
      total_num_blocks += num_blocks;
    }
    raft::update_device(workload_info, h_workload_info, total_num_blocks, handle.get_stream());
    return std::make_pair(total_num_blocks, n_large_nodes_in_curr_batch);
  }

  auto doSplit(const std::vector<NodeWorkItem>& work_items)
  {
    ML::PUSH_RANGE("Builder::doSplit @bulder_base.cuh [batched-levelalgo]");
    // auto batchSize = node_end - node_start;
    // start fresh on the number of *new* nodes created in this batch
    CUDA_CHECK(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), handle.get_stream()));
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, work_items.size(), handle.get_stream());

    // get the current set of nodes to be worked upon
    raft::update_device(d_work_items, work_items.data(), work_items.size(), handle.get_stream());

    auto [total_blocks, large_blocks] = this->updateWorkloadInfo(work_items);

    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    for (IdxT c = 0; c < input.nSampledCols; c += n_blks_for_cols) {
      computeSplit(c, work_items.size(), total_blocks, large_blocks);
      CUDA_CHECK(cudaGetLastError());
    }

    // create child nodes (or make the current ones leaf)
    auto smemSize = 2 * sizeof(IdxT) * TPB_DEFAULT;
    ML::PUSH_RANGE("nodeSplitKernel @builder_base.cuh [batched-levelalgo]");
    nodeSplitKernel<DataT, LabelT, IdxT, ObjectiveT, TPB_DEFAULT>
      <<<work_items.size(), TPB_DEFAULT, smemSize, handle.get_stream()>>>(
        params.max_depth,
        params.min_samples_leaf,
        params.min_samples_split,
        params.max_leaves,
        params.min_impurity_decrease,
        input,
        d_work_items,
        splits);
    CUDA_CHECK(cudaGetLastError());
    ML::POP_RANGE();
    raft::update_host(h_splits, splits, work_items.size(), handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    ML::POP_RANGE();
    return std::make_tuple(h_splits, work_items.size());
  }

  auto computeSplitSmemSize()
  {
    size_t smemSize1 = params.n_bins * input.numOutputs * sizeof(BinT) +  // pdf_shist size
                       params.n_bins * sizeof(DataT) +                    // sbins size
                       sizeof(int);                                       // sDone size
    // Extra room for alignment (see alignPointer in
    // computeSplitKernel)
    smemSize1 += sizeof(DataT) + 3 * sizeof(int);
    // Calculate the shared memory needed for evalBestSplit
    size_t smemSize2 = raft::ceildiv(TPB_DEFAULT, raft::WarpSize) * sizeof(SplitT);
    // Pick the max of two
    auto available_smem = handle.get_device_properties().sharedMemPerBlock;
    size_t smemSize     = std::max(smemSize1, smemSize2);
    ASSERT(available_smem >= smemSize, "Not enough shared memory. Consider reducing n_bins.");
    return smemSize;
  }

  void computeSplit(IdxT col, IdxT batchSize, size_t total_blocks, size_t large_blocks)
  {
    if (total_blocks == 0) return;
    ML::PUSH_RANGE("Builder::computeSplit @builder_base.cuh [batched-levelalgo]");
    auto nbins    = params.n_bins;
    auto nclasses = input.numOutputs;
    auto colBlks  = std::min(n_blks_for_cols, input.nSampledCols - col);

    auto smemSize = computeSplitSmemSize();
    dim3 grid(total_blocks, colBlks, 1);
    int nHistBins = large_blocks * nbins * colBlks * nclasses;
    CUDA_CHECK(cudaMemsetAsync(hist, 0, sizeof(BinT) * nHistBins, handle.get_stream()));
    ML::PUSH_RANGE("computeSplitClassificationKernel @builder_base.cuh [batched-levelalgo]");
    ObjectiveT objective(input.numOutputs, params.min_impurity_decrease, params.min_samples_leaf);
    computeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>
      <<<grid, TPB_DEFAULT, smemSize, handle.get_stream()>>>(hist,
                                                             params.n_bins,
                                                             params.max_depth,
                                                             params.min_samples_split,
                                                             params.max_leaves,
                                                             input,
                                                             d_work_items,
                                                             col,
                                                             done_count,
                                                             mutex,
                                                             splits,
                                                             objective,
                                                             treeid,
                                                             workload_info,
                                                             seed);
    ML::POP_RANGE();  // computeSplitKernel
    ML::POP_RANGE();  // Builder::computeSplit
  }

  // Set the leaf value predictions in batch
  void SetLeafPredictions(std::vector<NodeT>* tree)
  {
    // do this in batch to reduce peak memory usage in extreme cases
    std::size_t max_batch_size = min(std::size_t(100000), tree->size());
    rmm::device_uvector<NodeT> d_tree(max_batch_size, handle.get_stream());
    ObjectiveT objective(input.numOutputs, params.min_impurity_decrease, params.min_samples_leaf);
    for (std::size_t batch_begin = 0; batch_begin < tree->size(); batch_begin += max_batch_size) {
      std::size_t batch_end  = min(batch_begin + max_batch_size, tree->size());
      std::size_t batch_size = batch_end - batch_begin;
      raft::update_device(
        d_tree.data(), tree->data() + batch_begin, batch_size, handle.get_stream());
      size_t smemSize = sizeof(BinT) * input.numOutputs;
      int num_blocks  = batch_size;
      leafKernel<<<num_blocks, TPB_DEFAULT, smemSize, handle.get_stream()>>>(
        objective, input, d_tree.data());
      raft::update_host(tree->data() + batch_begin, d_tree.data(), batch_size, handle.get_stream());
    }
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  }
};  // end Builder

}  // namespace DT
}  // namespace ML