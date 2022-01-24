/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <memory>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include "kernels/builder_kernels.cuh"
#include <common/Timer.h>
#include <cuml/common/pinned_host_vector.hpp>
#include <cuml/tree/flatnode.h>
#include <raft/cuda_utils.cuh>

#include <deque>
#include <raft/common/nvtx.hpp>
#include <utility>

namespace ML {
namespace DT {

template <typename DataT, typename LabelT>
class NodeQueue {
  using NodeT = SparseTreeNode<DataT, LabelT>;
  const DecisionTreeParams params;
  std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> tree;
  std::vector<InstanceRange> node_instances_;
  std::deque<NodeWorkItem> work_items_;

 public:
  NodeQueue(DecisionTreeParams params, size_t max_nodes, size_t sampled_rows, int num_outputs)
    : params(params), tree(std::make_shared<DT::TreeMetaDataNode<DataT, LabelT>>())
  {
    tree->num_outputs = num_outputs;
    tree->sparsetree.reserve(max_nodes);
    tree->sparsetree.emplace_back(NodeT::CreateLeafNode(sampled_rows));
    tree->leaf_counter  = 1;
    tree->depth_counter = 0;
    node_instances_.reserve(max_nodes);
    node_instances_.emplace_back(InstanceRange{0, sampled_rows});
    if (this->IsExpandable(tree->sparsetree.back(), 0)) {
      work_items_.emplace_back(NodeWorkItem{0, 0, node_instances_.back()});
    }
  }

  std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> GetTree() { return tree; }
  const std::vector<InstanceRange>& GetInstanceRanges() { return node_instances_; }

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
  bool IsExpandable(const NodeT& n, int depth)
  {
    if (depth >= params.max_depth) return false;
    if (int(n.InstanceCount()) < params.min_samples_split) return false;
    if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) return false;
    return true;
  }

  template <typename SplitT>
  void Push(const std::vector<NodeWorkItem>& work_items, SplitT* h_splits)
  {
    // Update node queue based on splits
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto split        = h_splits[i];
      auto item         = work_items[i];
      auto parent_range = node_instances_.at(item.idx);
      if (SplitNotValid(
            split, params.min_impurity_decrease, params.min_samples_leaf, parent_range.count)) {
        continue;
      }

      if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) break;

      // parent
      tree->sparsetree.at(item.idx) = NodeT::CreateSplitNode(split.colid,
                                                             split.quesval,
                                                             split.best_metric_val,
                                                             int64_t(tree->sparsetree.size()),
                                                             parent_range.count);
      tree->leaf_counter++;
      // left
      tree->sparsetree.emplace_back(NodeT::CreateLeafNode(split.nLeft));
      node_instances_.emplace_back(InstanceRange{parent_range.begin, std::size_t(split.nLeft)});

      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, node_instances_.back()});
      }

      // right
      tree->sparsetree.emplace_back(NodeT::CreateLeafNode(parent_range.count - split.nLeft));
      node_instances_.emplace_back(
        InstanceRange{parent_range.begin + split.nLeft, parent_range.count - split.nLeft});

      // Do not add a work item if this child is definitely a leaf
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, node_instances_.back()});
      }

      // update depth
      tree->depth_counter = max(tree->depth_counter, item.depth + 1);
    }
  }
};

/**
 * Internal struct used to do all the heavy-lifting required for tree building
 */
template <typename ObjectiveT>
struct Builder {
  typedef typename ObjectiveT::DataT DataT;
  typedef typename ObjectiveT::LabelT LabelT;
  typedef typename ObjectiveT::IdxT IdxT;
  typedef typename ObjectiveT::BinT BinT;
  typedef SparseTreeNode<DataT, LabelT, IdxT> NodeT;
  typedef Split<DataT, IdxT> SplitT;
  typedef Input<DataT, LabelT, IdxT> InputT;

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 128;
  const raft::handle_t& handle;
  cudaStream_t builder_stream;
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
  std::shared_ptr<const rmm::device_uvector<DataT>> quantiles;

  WorkloadInfo<IdxT>* workload_info;
  WorkloadInfo<IdxT>* h_workload_info;

  int max_blocks = 0;

  SplitT* h_splits;
  /** number of blocks used to parallelize column-wise computations. */
  int n_blks_for_cols = 10;
  /** Memory alignment value */
  const size_t alignValue = 512;

  rmm::device_uvector<char> d_buff;
  ML::pinned_host_vector<char> h_buff;

  Builder(const raft::handle_t& handle,
          cudaStream_t s,
          IdxT treeid,
          uint64_t seed,
          const DecisionTreeParams& p,
          const DataT* data,
          const LabelT* labels,
          IdxT totalRows,
          IdxT totalCols,
          rmm::device_uvector<IdxT>* rowids,
          IdxT nclasses,
          std::shared_ptr<const rmm::device_uvector<DataT>> quantiles)
    : handle(handle),
      builder_stream(s),
      treeid(treeid),
      seed(seed),
      params(p),
      quantiles(quantiles),
      input{data,
            labels,
            totalRows,
            totalCols,
            int(rowids->size()),
            max(1, IdxT(params.max_features * totalCols)),
            rowids->data(),
            nclasses,
            quantiles->data()},
      d_buff(0, builder_stream)
  {
    max_blocks = 1 + params.max_batch_size + input.nSampledRows / TPB_DEFAULT;
    ASSERT(quantiles != nullptr, "Currently quantiles need to be computed before this call!");
    ASSERT(nclasses >= 1, "nclasses should be at least 1");

    auto [device_workspace_size, host_workspace_size] = workspaceSize();
    d_buff.resize(device_workspace_size, builder_stream);
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
    raft::common::nvtx::range fun_scope(
      "Builder::workspaceSize @builder_base.cuh [batched-levelalgo]");
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
    raft::common::nvtx::range fun_scope(
      "Builder::assignWorkspace @builder_base.cuh [batched-levelalgo]");
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

    RAFT_CUDA_TRY(
      cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, builder_stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, builder_stream));

    // host
    h_workload_info = reinterpret_cast<WorkloadInfo<IdxT>*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);
    h_splits = reinterpret_cast<SplitT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
  }

  std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> train()
  {
    raft::common::nvtx::range fun_scope("Builder::train @builder.cuh [batched-levelalgo]");
    MLCommon::TimerCPU timer;
    NodeQueue<DataT, LabelT> queue(params, this->maxNodes(), input.nSampledRows, input.numOutputs);
    while (queue.HasWork()) {
      auto work_items                      = queue.Pop();
      auto [splits_host_ptr, splits_count] = doSplit(work_items);
      queue.Push(work_items, splits_host_ptr);
    }
    auto tree = queue.GetTree();
    this->SetLeafPredictions(tree, queue.GetInstanceRanges());
    tree->train_time = timer.getElapsedMilliseconds();
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
      int num_blocks = raft::ceildiv(item.instances.count, size_t(TPB_DEFAULT));
      num_blocks     = std::max(1, num_blocks);

      if (num_blocks > 1) ++n_large_nodes_in_curr_batch;

      for (int b = 0; b < num_blocks; b++) {
        h_workload_info[total_num_blocks + b] = {
          int(i), n_large_nodes_in_curr_batch - 1, b, num_blocks};
      }
      total_num_blocks += num_blocks;
    }
    raft::update_device(workload_info, h_workload_info, total_num_blocks, builder_stream);
    return std::make_pair(total_num_blocks, n_large_nodes_in_curr_batch);
  }

  auto doSplit(const std::vector<NodeWorkItem>& work_items)
  {
    raft::common::nvtx::range fun_scope("Builder::doSplit @bulder_base.cuh [batched-levelalgo]");
    // start fresh on the number of *new* nodes created in this batch
    RAFT_CUDA_TRY(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), builder_stream));
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, work_items.size(), builder_stream);

    // get the current set of nodes to be worked upon
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);

    auto [total_blocks, large_blocks] = this->updateWorkloadInfo(work_items);

    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    for (IdxT c = 0; c < input.nSampledCols; c += n_blks_for_cols) {
      computeSplit(c, work_items.size(), total_blocks, large_blocks);
      RAFT_CUDA_TRY(cudaGetLastError());
    }

    // create child nodes (or make the current ones leaf)
    auto smemSize = 2 * sizeof(IdxT) * TPB_DEFAULT;
    raft::common::nvtx::push_range("nodeSplitKernel @builder_base.cuh [batched-levelalgo]");
    nodeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>
      <<<work_items.size(), TPB_DEFAULT, smemSize, builder_stream>>>(params.max_depth,
                                                                     params.min_samples_leaf,
                                                                     params.min_samples_split,
                                                                     params.max_leaves,
                                                                     params.min_impurity_decrease,
                                                                     input,
                                                                     d_work_items,
                                                                     splits);
    RAFT_CUDA_TRY(cudaGetLastError());
    raft::common::nvtx::pop_range();
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(builder_stream));
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
    raft::common::nvtx::range fun_scope(
      "Builder::computeSplit @builder_base.cuh [batched-levelalgo]");
    auto nbins    = params.n_bins;
    auto nclasses = input.numOutputs;
    auto colBlks  = std::min(n_blks_for_cols, input.nSampledCols - col);

    auto smemSize = computeSplitSmemSize();
    dim3 grid(total_blocks, colBlks, 1);
    int nHistBins = large_blocks * nbins * colBlks * nclasses;
    RAFT_CUDA_TRY(cudaMemsetAsync(hist, 0, sizeof(BinT) * nHistBins, builder_stream));
    raft::common::nvtx::range kernel_scope(
      "computeSplitClassificationKernel @builder_base.cuh [batched-levelalgo]");
    ObjectiveT objective(input.numOutputs, params.min_samples_leaf);
    computeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>
      <<<grid, TPB_DEFAULT, smemSize, builder_stream>>>(hist,
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
  }

  // Set the leaf value predictions in batch
  void SetLeafPredictions(std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> tree,
                          const std::vector<InstanceRange>& instance_ranges)
  {
    tree->vector_leaf.resize(tree->sparsetree.size() * input.numOutputs);
    ASSERT(tree->sparsetree.size() == instance_ranges.size(),
           "Expected instance range for each node");
    // do this in batch to reduce peak memory usage in extreme cases
    std::size_t max_batch_size = min(std::size_t(100000), tree->sparsetree.size());
    rmm::device_uvector<NodeT> d_tree(max_batch_size, builder_stream);
    rmm::device_uvector<InstanceRange> d_instance_ranges(max_batch_size, builder_stream);
    rmm::device_uvector<DataT> d_leaves(max_batch_size * input.numOutputs, builder_stream);

    ObjectiveT objective(input.numOutputs, params.min_samples_leaf);
    for (std::size_t batch_begin = 0; batch_begin < tree->sparsetree.size();
         batch_begin += max_batch_size) {
      std::size_t batch_end  = min(batch_begin + max_batch_size, tree->sparsetree.size());
      std::size_t batch_size = batch_end - batch_begin;
      raft::update_device(
        d_tree.data(), tree->sparsetree.data() + batch_begin, batch_size, builder_stream);
      raft::update_device(
        d_instance_ranges.data(), instance_ranges.data() + batch_begin, batch_size, builder_stream);

      RAFT_CUDA_TRY(
        cudaMemsetAsync(d_leaves.data(), 0, sizeof(DataT) * d_leaves.size(), builder_stream));
      size_t smemSize = sizeof(BinT) * input.numOutputs;
      int num_blocks  = batch_size;
      leafKernel<<<num_blocks, TPB_DEFAULT, smemSize, builder_stream>>>(
        objective, input, d_tree.data(), d_instance_ranges.data(), d_leaves.data());
      raft::update_host(tree->vector_leaf.data() + batch_begin * input.numOutputs,
                        d_leaves.data(),
                        batch_size * input.numOutputs,
                        builder_stream);
    }
  }
};  // end Builder

}  // namespace DT
}  // namespace ML
