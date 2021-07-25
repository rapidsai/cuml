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
#include <cuml/common/device_buffer.hpp>
#include <cuml/common/host_buffer.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <raft/cuda_utils.cuh>
#include "cuml/common/logger.hpp"
#include "input.cuh"
#include "kernels.cuh"
#include "metrics.cuh"
#include "node.cuh"
#include "split.cuh"

#include <common/nvtx.hpp>
#include <utility>

namespace ML {
namespace DT {

template <typename NodeT>
class NodeQueue {
  std::vector<NodeT> queue_;
  int tree_depth_ = 0;
  int num_leaves_ = 0;
  size_t active_range_begin_;

 public:
  NodeQueue(size_t max_nodes, size_t sampled_rows)
  {
    queue_.reserve(max_nodes);
    queue_.emplace_back(0, sampled_rows, 0);
    active_range_begin_ = 0;
  }

  int TreeDepth() { return tree_depth_; }
  int NumLeaves() { return num_leaves_; }
  std::vector<NodeT> GetTree() { return queue_; }

  bool HasWork() { return active_range_begin_ < queue_.size(); }
  
  // Returns pointer and length containing active set of nodes
  auto Pop(size_t max_batch_size)
  {
    ASSERT(active_range_begin_ <= queue_.size(), "Node queue error.");
    auto num_popped     = std::min(max_batch_size, queue_.size() - active_range_begin_);
    auto result         = std::make_pair(queue_.data() + active_range_begin_, num_popped);
    active_range_begin_ = active_range_begin_ + num_popped;
    return result;
  }

  template <typename SplitT, typename LabelT>
  void Push(DecisionTreeParams params,
            SplitT* h_splits,
            size_t num_splits,
            LabelT* h_predictions,
            size_t num_predictions)
  {
    // Update node queue based on splits
    ASSERT(num_splits <= active_range_begin_, "Node queue error.");
    for (int i = 0; i < num_splits; i++) {
      auto split = h_splits[i];
      // Splits refers to the previous active range of nodes
      size_t node_idx = active_range_begin_ - num_splits + i;
      const auto node = queue_[active_range_begin_ - num_splits + i];
      auto isLeaf =
        leafBasedOnParams<int>(node.depth, params.max_depth, params.min_samples_split, node.count);
      if (isLeaf || split.best_metric_val <= params.min_impurity_decrease ||
          split.nLeft < params.min_samples_leaf ||
          (node.count - split.nLeft) < params.min_samples_leaf) {
        // Set leaf
        queue_[node_idx].makeLeaf(h_predictions[i]);
        num_leaves_++;
      } else {
        // Make children

        // parent
        queue_[node_idx] =
          NodeT::CreateSplit(split.colid, split.quesval, split.best_metric_val, queue_.size());
        // left
        queue_.emplace_back(
          NodeT::CreateChild(node.depth + 1, node.start, split.nLeft, 2 * node.info.unique_id + 1));
        // right
        queue_.emplace_back(NodeT::CreateChild(node.depth + 1,
                                               node.start + split.nLeft,
                                               node.count - split.nLeft,
                                               2 * node.info.unique_id + 2));
        // update depth
        tree_depth_ = max(tree_depth_, node.depth + 1);
      }
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
  /** threads per block for the nodeSplitKernel */
  static constexpr int TPB_SPLIT = 128;
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
  NodeT* curr_nodes;
  LabelT* predictions;

  WorkloadInfo<IdxT>* workload_info;
  WorkloadInfo<IdxT>* h_workload_info;

  int max_blocks = 0;

  SplitT* h_splits;
  LabelT* h_predictions;
  /** number of blocks used to parallelize column-wise computations. */
  int n_blks_for_cols = 10;
  /** Memory alignment value */
  const size_t alignValue = 512;

  MLCommon::device_buffer<char> d_buff;
  MLCommon::host_buffer<char> h_buff;

  Builder(const raft::handle_t& handle,
          IdxT treeid,
          uint64_t seed,
          const DecisionTreeParams& p,
          const DataT* data,
          const LabelT* labels,
          IdxT totalRows,
          IdxT totalCols,
          IdxT sampledRows,
          IdxT sampledCols,
          IdxT* rowids,
          IdxT nclasses,
          const DataT* quantiles)
    : handle(handle),
      treeid(treeid),
      seed(seed),
      params(p),
      input{
        data, labels, totalRows, totalCols, sampledRows, sampledCols, rowids, nclasses, quantiles},
      d_buff(handle.get_device_allocator(), handle.get_stream(), 0),
      h_buff(handle.get_host_allocator(), handle.get_stream(), 0)
  {
    max_blocks = 1 + params.max_batch_size + input.nSampledRows / TPB_DEFAULT;
    ASSERT(quantiles != nullptr, "Currently quantiles need to be computed before this call!");
    ASSERT(nclasses >= 1, "nclasses should be at least 1");

    auto [device_workspace_size, host_workspace_size] = workspaceSize();
    d_buff.resize(device_workspace_size, handle.get_stream());
    h_buff.resize(host_workspace_size, handle.get_stream());
    assignWorkspace(d_buff.data(), h_buff.data());
  }
  ~Builder()
  {
    d_buff.release(handle.get_stream());
    h_buff.release(handle.get_stream());
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
    d_wsize += calculateAlignedBytes(sizeof(NodeT) * max_batch);                  // curr_nodes
    d_wsize += calculateAlignedBytes(sizeof(NodeT) * 2 * max_batch);              // next_nodes
    d_wsize +=                                                                    // workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);
    d_wsize += calculateAlignedBytes(sizeof(LabelT) * max_batch);  // predictions

    // all nodes in the tree
    h_wsize +=  // h_workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);
    h_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);  // splits

    h_wsize += calculateAlignedBytes(sizeof(LabelT) * max_batch);  // predictions

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
    curr_nodes = reinterpret_cast<NodeT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(NodeT) * max_batch);
    workload_info = reinterpret_cast<WorkloadInfo<IdxT>*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);
    predictions = reinterpret_cast<LabelT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(LabelT) * max_batch);

    CUDA_CHECK(
      cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, handle.get_stream()));
    CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, handle.get_stream()));

    // host
    h_workload_info = reinterpret_cast<WorkloadInfo<IdxT>*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);
    h_splits = reinterpret_cast<SplitT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    h_predictions = reinterpret_cast<LabelT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(LabelT) * max_batch);
    ML::POP_RANGE();
  }

  std::vector<Node<DataT, LabelT, IdxT>> train(IdxT& num_leaves, IdxT& depth, cudaStream_t s)
  {
    ML::PUSH_RANGE("Builder::train @builder_base.cuh [batched-levelalgo]");
    NodeQueue<NodeT> queue(this->maxNodes(), input.nSampledRows);
    while (queue.HasWork()) {
      auto [new_nodes_host_ptr, new_nodes_count] = queue.Pop(params.max_batch_size);
      auto [splits_host_ptr, splits_count, predictions_host_ptr, predictions_count] =
        doSplit(new_nodes_host_ptr, new_nodes_count, s);
      queue.Push(params, splits_host_ptr, splits_count, predictions_host_ptr, predictions_count);
    }
    depth      = queue.TreeDepth();
    num_leaves = queue.NumLeaves();
    ML::POP_RANGE();
    return queue.GetTree();
  }

  size_t nodeSplitSmemSize()
  {
    return std::max(2 * sizeof(IdxT) * TPB_SPLIT, sizeof(BinT) * input.numOutputs);
  }

 private:
  auto updateWorkloadInfo(const Node<DataT, LabelT, IdxT>* h_nodes,
                          size_t batchSize,
                          cudaStream_t s)
  {
    size_t total_samples_in_curr_batch = 0;
    size_t n_large_nodes_in_curr_batch =
      0;  // large nodes are nodes having training instances larger than block size, hence require
          // global memory for histogram construction
    size_t total_num_blocks = 0;
    for (int n = 0; n < batchSize; n++) {
      total_samples_in_curr_batch += h_nodes[n].count;
      size_t num_blocks = raft::ceildiv(h_nodes[n].count, TPB_DEFAULT);
      num_blocks        = std::max(size_t(1), num_blocks);

      if (num_blocks > 1) ++n_large_nodes_in_curr_batch;

      bool is_leaf = leafBasedOnParams<IdxT>(
        h_nodes[n].depth, params.max_depth, params.min_samples_split, h_nodes[n].count);
      if (is_leaf) num_blocks = 0;

      for (int b = 0; b < num_blocks; b++) {
        h_workload_info[total_num_blocks + b].nodeid         = n;
        h_workload_info[total_num_blocks + b].large_nodeid   = n_large_nodes_in_curr_batch - 1;
        h_workload_info[total_num_blocks + b].offset_blockid = b;
        h_workload_info[total_num_blocks + b].num_blocks     = num_blocks;
      }
      total_num_blocks += num_blocks;
    }
    raft::update_device(workload_info, h_workload_info, total_num_blocks, s);
    return std::make_pair(total_num_blocks, n_large_nodes_in_curr_batch);
  }
  /**
   * @brief Computes best split across all nodes in the current batch and splits
   *        the nodes accordingly
   *
   * @param[out] h_nodes list of nodes (must be allocated using cudaMallocHost!)
   * @param[in]  s cuda stream
   * @return the number of newly created nodes
   */
  auto doSplit(const Node<DataT, LabelT, IdxT>* h_nodes, size_t batchSize, cudaStream_t s)
  {
    ML::PUSH_RANGE("Builder::doSplit @bulder_base.cuh [batched-levelalgo]");
    // auto batchSize = node_end - node_start;
    // start fresh on the number of *new* nodes created in this batch
    CUDA_CHECK(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), s));
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, batchSize, s);

    // get the current set of nodes to be worked upon
    raft::update_device(curr_nodes, h_nodes, batchSize, s);

    auto [total_blocks, large_blocks] = this->updateWorkloadInfo(h_nodes, batchSize, s);

    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    for (IdxT c = 0; c < input.nSampledCols; c += n_blks_for_cols) {
      computeSplit(c, batchSize, params.split_criterion, total_blocks, large_blocks, s);
      CUDA_CHECK(cudaGetLastError());
    }

    // create child nodes (or make the current ones leaf)
    auto smemSize = nodeSplitSmemSize();
    ML::PUSH_RANGE("nodeSplitKernel @builder_base.cuh [batched-levelalgo]");
    nodeSplitKernel<DataT, LabelT, IdxT, ObjectiveT, TPB_SPLIT>
      <<<batchSize, TPB_SPLIT, smemSize, s>>>(params.max_depth,
                                              params.min_samples_leaf,
                                              params.min_samples_split,
                                              params.max_leaves,
                                              params.min_impurity_decrease,
                                              input,
                                              curr_nodes,
                                              splits,
                                              predictions);
    CUDA_CHECK(cudaGetLastError());
    ML::POP_RANGE();
    raft::update_host(h_splits, splits, batchSize, s);
    raft::update_host(h_predictions, predictions, batchSize, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    ML::POP_RANGE();
    return std::make_tuple(h_splits, batchSize, h_predictions, batchSize);
  }

  auto computeSplitSmemSize()
  {
    size_t smemSize1 = params.n_bins * input.numOutputs * sizeof(BinT) +  // pdf_shist size
                       params.n_bins * sizeof(DataT) +                    // sbins size
                       sizeof(int);                                       // sDone size
    // Extra room for alignment (see alignPointer in
    // computeSplitClassificationKernel)
    smemSize1 += sizeof(DataT) + 3 * sizeof(int);
    // Calculate the shared memory needed for evalBestSplit
    size_t smemSize2 = raft::ceildiv(TPB_DEFAULT, raft::WarpSize) * sizeof(SplitT);
    // Pick the max of two
    auto available_smem = handle.get_device_properties().sharedMemPerBlock;
    size_t smemSize     = std::max(smemSize1, smemSize2);
    ASSERT(available_smem >= smemSize, "Not enough shared memory. Consider reducing n_bins.");
    return smemSize;
  }

  /**
   * @brief Compute best split for the currently given set of columns
   *
   * @param[in] col       start column id
   * @param[in] batchSize number of nodes to be processed in this call
   * @param[in] splitType split criterion
   * @param[in] s         cuda stream
   */
  void computeSplit(IdxT col,
                    IdxT batchSize,
                    CRITERION splitType,
                    size_t total_blocks,
                    size_t large_blocks,
                    cudaStream_t s)
  {
    if (total_blocks == 0) return;
    ML::PUSH_RANGE("Builder::computeSplit @builder_base.cuh [batched-levelalgo]");
    auto nbins    = params.n_bins;
    auto nclasses = input.numOutputs;
    auto colBlks  = std::min(n_blks_for_cols, input.nSampledCols - col);

    auto smemSize = computeSplitSmemSize();
    dim3 grid(total_blocks, colBlks, 1);
    int nHistBins = large_blocks * nbins * colBlks * nclasses;
    CUDA_CHECK(cudaMemsetAsync(hist, 0, sizeof(BinT) * nHistBins, s));
    ML::PUSH_RANGE("computeSplitClassificationKernel @builder_base.cuh [batched-levelalgo]");
    ObjectiveT objective(input.numOutputs, params.min_impurity_decrease, params.min_samples_leaf);
    computeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>
      <<<grid, TPB_DEFAULT, smemSize, s>>>(hist,
                                           params.n_bins,
                                           params.max_depth,
                                           params.min_samples_split,
                                           params.max_leaves,
                                           input,
                                           curr_nodes,
                                           col,
                                           done_count,
                                           mutex,
                                           splits,
                                           objective,
                                           treeid,
                                           workload_info,
                                           seed);
    ML::POP_RANGE();  // computeSplitClassificationKernel
    ML::POP_RANGE();  // Builder::computeSplit
  }
};  // end Builder

}  // namespace DT
}  // namespace ML
