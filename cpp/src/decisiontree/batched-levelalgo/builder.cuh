/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels/builder_kernels.cuh"

#include <common/Timer.h>

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/pinned_host_vector.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <algorithm>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

namespace ML {
namespace DT {

/**
 * Structure that manages the iterative batched-level training and building of nodes
 * in the host.
 */
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
  typedef Dataset<DataT, LabelT, IdxT> DatasetT;
  typedef Quantiles<DataT, IdxT> QuantilesT;

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 128;
  // Tunable performance heuristic for the shared-memory histogram path. Large per-block
  // histograms, usually from large n_classes, can reduce occupancy enough that global memory is
  // faster even when the histogram fits in shared memory. 16 KiB keeps small/default histograms in
  // shared memory while avoiding the large-class shared-memory slowdown measured locally.
  static constexpr size_t tunable_split_histogram_dynamic_smem_limit_bytes = 16 * 1024;
  /** handle to get device properties */
  const raft::handle_t& handle;
  /** stream to launch kernels */
  cudaStream_t builder_stream;
  /** DT params */
  DecisionTreeParams params;
  /** input dataset */
  DatasetT dataset;
  /** quantiles */
  QuantilesT quantiles;
  /** Tree index */
  IdxT treeid;
  /** Seed used for randomization */
  uint64_t seed;
  /** number of nodes created in the current batch */
  IdxT* n_nodes;
  /** buffer of segmented histograms*/
  BinT* histograms;
  /** threadblock arrival count */
  int* done_count;
  /** mutex array used for atomically updating best split */
  int* mutex;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  NodeWorkItem* d_work_items;
  /** device AOS to map CTA blocks along dimx to nodes of a batch */
  WorkloadInfo<IdxT>* workload_info;
  /** host AOS to map CTA blocks along dimx to nodes of a batch */
  WorkloadInfo<IdxT>* h_workload_info;
  /** maximum CTA blocks along dimx */
  int max_blocks_dimx = 0;
  /** host array of splits */
  SplitT* h_splits;
  /** number of blocks used to parallelize column-wise computations */
  int n_blks_for_cols = 10;
  /** Memory alignment value */
  const size_t align_value = 512;
  IdxT* column_samples;
  /** temporary row IDs for row-wise out-of-place partitioning */
  IdxT* partition_row_ids;
  /** rmm device workspace buffer */
  rmm::device_uvector<char> d_buff;
  /** pinned host buffer to store the trained nodes */
  ML::pinned_host_vector<char> h_buff;

  Builder(const raft::handle_t& handle,
          cudaStream_t s,
          IdxT treeid,
          uint64_t seed,
          const DecisionTreeParams& p,
          const DataT* data,
          const LabelT* labels,
          const double* sample_weight,
          IdxT n_rows,
          IdxT n_cols,
          rmm::device_uvector<IdxT>* row_ids,
          IdxT n_classes,
          const QuantilesT& q)
    : handle(handle),
      builder_stream(s),
      treeid(treeid),
      seed(seed),
      params(p),
      dataset{data,
              labels,
              sample_weight,
              n_rows,
              n_cols,
              int(row_ids->size()),
              max(1, IdxT(params.max_features * n_cols)),
              row_ids->data(),
              n_classes},
      quantiles(q),
      d_buff(0, builder_stream)
  {
    max_blocks_dimx = 1 + params.max_batch_size + dataset.n_sampled_rows / TPB_DEFAULT;
    ASSERT(q.quantiles_array != nullptr && q.n_bins_array != nullptr,
           "Currently quantiles need to be computed before this call!");
    ASSERT(n_classes >= 1, "n_classes should be at least 1");

    auto [device_workspace_size, host_workspace_size] = workspaceSize();
    d_buff.resize(device_workspace_size, builder_stream);
    h_buff.resize(host_workspace_size);
    assignWorkspace(d_buff.data(), h_buff.data());
  }

  /**
   * @brief calculates nearest aligned size of input w.r.t an `align_value`.
   *
   * @param[in] actual_size actual size in bytes of input
   * @return aligned size
   */
  size_t calculateAlignedBytes(const size_t actual_size) const
  {
    return raft::alignTo(actual_size, align_value);
  }

  /**
   * @brief returns maximum nodes possible per tree
   * @return maximum nodes possible per tree
   */
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

  /**
   * @brief calculate the workspace size required
   *
   * @return a pair of device workspace and host workspace size requirements
   */
  auto workspaceSize() const
  {
    size_t d_wsize = 0, h_wsize = 0;
    raft::common::nvtx::range fun_scope("Builder::workspaceSize @builder.cuh [batched-levelalgo]");
    auto max_batch = params.max_batch_size;
    size_t max_len_histograms =
      max_batch * params.max_n_bins * n_blks_for_cols * dataset.num_outputs;

    d_wsize += calculateAlignedBytes(sizeof(IdxT));                               // n_nodes
    d_wsize += calculateAlignedBytes(sizeof(BinT) * max_len_histograms);          // histograms
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch * n_blks_for_cols);  // done_count
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch);                    // mutex
    d_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);                 // splits
    d_wsize += calculateAlignedBytes(sizeof(NodeWorkItem) * max_batch);           // d_work_Items
    d_wsize +=                                                                    // workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks_dimx);
    d_wsize +=
      calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);  // column_samples
    d_wsize += calculateAlignedBytes(sizeof(IdxT) * dataset.n_sampled_rows);  // partition row IDs

    // all nodes in the tree
    h_wsize +=  // h_workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks_dimx);
    h_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);  // splits

    return std::make_pair(d_wsize, h_wsize);
  }

  /**
   * @brief assign workspace to the current state
   *
   * @param[in] d_wspace device buffer allocated by the user for the workspace.
   *                     Its size should be at least workspaceSize()
   * @param[in] h_wspace pinned host buffer needed to store the learned nodes
   */
  void assignWorkspace(char* d_wspace, char* h_wspace)
  {
    raft::common::nvtx::range fun_scope(
      "Builder::assignWorkspace @builder.cuh [batched-levelalgo]");
    auto max_batch  = params.max_batch_size;
    auto n_col_blks = n_blks_for_cols;
    size_t max_len_histograms =
      max_batch * (params.max_n_bins) * n_blks_for_cols * dataset.num_outputs;
    // device
    n_nodes = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT));
    histograms = reinterpret_cast<BinT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(BinT) * max_len_histograms);
    done_count = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(int) * max_batch * n_col_blks);
    mutex = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(int) * max_batch);
    splits = reinterpret_cast<SplitT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    d_work_items = reinterpret_cast<NodeWorkItem*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(NodeWorkItem) * max_batch);
    workload_info = reinterpret_cast<WorkloadInfo<IdxT>*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks_dimx);
    column_samples = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);
    partition_row_ids = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT) * dataset.n_sampled_rows);

    RAFT_CUDA_TRY(
      cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, builder_stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, builder_stream));

    // host
    h_workload_info = reinterpret_cast<WorkloadInfo<IdxT>*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks_dimx);
    h_splits = reinterpret_cast<SplitT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
  }

  /**
   * @brief trains the tree, builds the nodes
   *
   * @return trained tree structure
   */
  std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> train()
  {
    raft::common::nvtx::range fun_scope("Builder::train @builder.cuh [batched-levelalgo]");
    MLCommon::TimerCPU timer;
    NodeQueue<DataT, LabelT> queue(
      params, this->maxNodes(), dataset.n_sampled_rows, dataset.num_outputs);
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
    int n_large_nodes = 0;  // large nodes are nodes having training instances larger than block
                            // size, hence require global memory for histogram construction
    int n_blocks_dimx = 0;  // gridDim.x required for computeSplitKernel
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto item = work_items[i];
      int n_blocks_per_node =
        std::max(raft::ceildiv(item.instances.count, size_t(TPB_DEFAULT)), size_t(1));

      if (n_blocks_per_node > 1) ++n_large_nodes;

      for (int b = 0; b < n_blocks_per_node; b++) {
        h_workload_info[n_blocks_dimx + b] = {int(i), n_large_nodes - 1, b, n_blocks_per_node};
      }
      n_blocks_dimx += n_blocks_per_node;
    }
    raft::update_device(workload_info, h_workload_info, n_blocks_dimx, builder_stream);
    return std::make_pair(n_blocks_dimx, n_large_nodes);
  }

  auto doSplit(const std::vector<NodeWorkItem>& work_items)
  {
    raft::common::nvtx::range fun_scope("Builder::doSplit @builder.cuh [batched-levelalgo]");
    // start fresh on the number of *new* nodes created in this batch
    RAFT_CUDA_TRY(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), builder_stream));

    const IdxT original_n_sampled_cols = dataset.n_sampled_cols;
    ASSERT(original_n_sampled_cols > 0 && original_n_sampled_cols <= dataset.N,
           "n_sampled_cols must be in [1, n_cols]");
    const std::size_t max_sampling_rounds =
      std::size_t((dataset.N + original_n_sampled_cols - 1) / original_n_sampled_cols);
    struct HostSplit {
      DataT quesval;
      IdxT colid;
      DataT best_metric_val;
      int nLeft;
      IdxT split_start;
      IdxT split_end;
    };
    static_assert(sizeof(HostSplit) == sizeof(SplitT));
    static_assert(alignof(HostSplit) == alignof(SplitT));

    // The final split chosen for each original work item. Nodes that need
    // additional feature samples are compacted in active_items, so successful
    // splits must be copied back to their original batch position.
    std::vector<HostSplit> final_splits(work_items.size());
    // Current retry batch. It starts as the full batch and shrinks to only
    // nodes whose sampled features did not produce a valid split.
    std::vector<NodeWorkItem> active_items(work_items);
    // active_items[i] maps back to the corresponding index in the original
    // work_items/final_splits arrays.
    std::vector<std::size_t> active_to_original(work_items.size());
    for (std::size_t i = 0; i < active_to_original.size(); ++i) {
      active_to_original[i] = i;
    }

    // Match sklearn's behavior of searching beyond max_features when the
    // sampled features do not yield a valid split.
    for (std::size_t round = 0; !active_items.empty() && round < max_sampling_rounds; ++round) {
      IdxT sample_offset     = IdxT(round) * original_n_sampled_cols;
      dataset.n_sampled_cols = std::min(original_n_sampled_cols, dataset.N - sample_offset);
      computeBestSplits(active_items, seed, sample_offset);

      std::vector<NodeWorkItem> retry_items;
      std::vector<std::size_t> retry_to_original;
      for (std::size_t i = 0; i < active_items.size(); ++i) {
        const auto original_idx    = active_to_original[i];
        final_splits[original_idx] = HostSplit{h_splits[i].quesval,
                                               h_splits[i].colid,
                                               h_splits[i].best_metric_val,
                                               h_splits[i].nLeft,
                                               h_splits[i].split_start,
                                               h_splits[i].split_end};
        if (SplitPartitionNotValid(
              h_splits[i], params.min_samples_leaf, active_items[i].instances.count)) {
          retry_items.push_back(active_items[i]);
          retry_to_original.push_back(original_idx);
        }
      }

      if (round + 1 >= max_sampling_rounds) { break; }
      active_items       = std::move(retry_items);
      active_to_original = std::move(retry_to_original);
    }
    dataset.n_sampled_cols = original_n_sampled_cols;

    // Partition samples once, using the valid split found for each node. Nodes
    // still without a valid split after all features have been visited remain leaves.
    RAFT_CUDA_TRY(cudaMemcpyAsync(splits,
                                  final_splits.data(),
                                  sizeof(SplitT) * work_items.size(),
                                  cudaMemcpyHostToDevice,
                                  builder_stream));
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);
    const auto partition_workload = this->updateWorkloadInfo(work_items);
    raft::common::nvtx::push_range("nodeSplitKernel @builder.cuh [batched-levelalgo]");
    launchNodeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>(params.min_samples_leaf,
                                                            params.min_impurity_decrease,
                                                            dataset,
                                                            d_work_items,
                                                            splits,
                                                            workload_info,
                                                            partition_workload.first,
                                                            partition_row_ids,
                                                            builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::common::nvtx::pop_range();
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    handle.sync_stream(builder_stream);
    return std::make_tuple(h_splits, work_items.size());
  }

  void computeBestSplits(const std::vector<NodeWorkItem>& work_items,
                         uint64_t sampling_seed,
                         IdxT sample_offset)
  {
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, work_items.size(), builder_stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      done_count, 0, sizeof(int) * params.max_batch_size * n_blks_for_cols, builder_stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, sizeof(int) * params.max_batch_size, builder_stream));
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);
    auto [n_blocks_dimx, n_large_nodes] = this->updateWorkloadInfo(work_items);
    auto split_smem_config              = computeSplitSharedMemoryConfig();

    sampleFeatures(work_items, sampling_seed, sample_offset);

    for (IdxT c = 0; c < dataset.n_sampled_cols; c += n_blks_for_cols) {
      computeSplit(c, n_blocks_dimx, n_large_nodes, work_items.size(), split_smem_config);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    handle.sync_stream(builder_stream);
  }

  void sampleFeatures(const std::vector<NodeWorkItem>& work_items,
                      uint64_t sampling_seed,
                      IdxT sample_offset)
  {
    raft::common::nvtx::range fun_scope("feature-sampling");
    sample_features<IdxT>(column_samples,
                          d_work_items,
                          work_items.size(),
                          treeid,
                          sampling_seed,
                          sample_offset,
                          dataset.N,
                          dataset.n_sampled_cols,
                          builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  struct SplitSharedMemoryConfig {
    bool use_global_memory_histogram;
    size_t dynamic_smem_size;
  };

  SplitSharedMemoryConfig computeSplitSharedMemoryConfig() const
  {
    // Dynamic shared memory for the fast path: histogram, copied quantiles, and
    // alignment padding for the kernel's shared-memory layout.
    auto shared_histogram_size =
      ML::checked_mul<std::size_t>(params.max_n_bins, dataset.num_outputs, sizeof(BinT));
    auto shared_quantiles_size = ML::checked_mul<std::size_t>(params.max_n_bins, sizeof(DataT));
    auto shared_dynamic_smem_size =
      ML::checked_add<std::size_t>(shared_histogram_size, shared_quantiles_size, sizeof(int));
    auto alignment_smem_size =
      ML::checked_add<std::size_t>(sizeof(DataT), ML::checked_mul<std::size_t>(3, sizeof(int)));
    shared_dynamic_smem_size =
      ML::checked_add<std::size_t>(shared_dynamic_smem_size, alignment_smem_size);

    // Dynamic shared memory for the fallback path only needs the per-block done
    // flag used by the inter-block completion handshake.
    auto global_dynamic_smem_size = ML::checked_add<std::size_t>(sizeof(int), sizeof(int));

    // Static shared memory is reserved by the kernel regardless of where the
    // histogram lives.
    auto cdf_scan_smem_size = sizeof(typename cub::BlockScan<BinT, TPB_DEFAULT>::TempStorage);
    auto split_scratch_smem_size =
      ML::checked_mul<std::size_t>(raft::ceildiv(TPB_DEFAULT, raft::WarpSize), sizeof(SplitT));
    auto static_smem_size =
      ML::checked_add<std::size_t>(cdf_scan_smem_size, split_scratch_smem_size);

    auto available_smem = size_t(handle.get_device_properties().sharedMemPerBlock);
    auto global_total_smem_size =
      ML::checked_add<std::size_t>(global_dynamic_smem_size, static_smem_size);
    ASSERT(available_smem >= global_total_smem_size,
           "Not enough shared memory for RF split bookkeeping.");

    // Prefer shared memory when it fits and stays small enough for good occupancy;
    // otherwise use the global histogram path to avoid launch failure or slowdown.
    auto shared_total_smem_size =
      ML::checked_add<std::size_t>(shared_dynamic_smem_size, static_smem_size);
    bool use_global_memory_histogram =
      shared_total_smem_size > available_smem ||
      shared_dynamic_smem_size > tunable_split_histogram_dynamic_smem_limit_bytes;

    return {use_global_memory_histogram,
            use_global_memory_histogram ? global_dynamic_smem_size : shared_dynamic_smem_size};
  }

  void computeSplit(IdxT col,
                    size_t n_blocks_dimx,
                    size_t n_large_nodes,
                    size_t n_work_items,
                    const SplitSharedMemoryConfig& split_smem_config)
  {
    // if no instances to split, return
    if (n_blocks_dimx == 0) return;
    raft::common::nvtx::range fun_scope("Builder::computeSplit @builder.cuh [batched-levelalgo]");
    auto n_bins                      = params.max_n_bins;
    auto n_classes                   = dataset.num_outputs;
    auto use_global_memory_histogram = split_smem_config.use_global_memory_histogram;
    // if columns left to be processed lesser than `n_blks_for_cols`, shrink the blocks along dimy
    auto n_blocks_dimy = std::min(n_blks_for_cols, dataset.n_sampled_cols - col);
    dim3 grid(n_blocks_dimx, n_blocks_dimy, 1);
    auto histogram_node_count = use_global_memory_histogram ? n_work_items : n_large_nodes;
    auto len_histograms =
      ML::checked_mul<std::size_t>(n_bins, n_classes, n_blocks_dimy, histogram_node_count);
    auto histograms_bytes = ML::checked_mul<std::size_t>(sizeof(BinT), len_histograms);
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms, 0, histograms_bytes, builder_stream));
    // create the objective function object
    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf, params.split_criterion);
    // call the computeSplitKernel
    raft::common::nvtx::range kernel_scope("computeSplitKernel @builder.cuh [batched-levelalgo]");
    launchComputeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT, ObjectiveT>(
      histograms,
      params.max_n_bins,
      params.min_samples_split,
      params.max_leaves,
      dataset,
      quantiles,
      d_work_items,
      col,
      column_samples,
      done_count,
      mutex,
      splits,
      objective,
      treeid,
      workload_info,
      seed,
      use_global_memory_histogram,
      grid,
      split_smem_config.dynamic_smem_size,
      builder_stream);
  }

  // Set the leaf value predictions in batch
  void SetLeafPredictions(std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> tree,
                          const std::vector<InstanceRange>& instance_ranges)
  {
    tree->vector_leaf.resize(tree->sparsetree.size() * dataset.num_outputs);
    ASSERT(tree->sparsetree.size() == instance_ranges.size(),
           "Expected instance range for each node");
    // do this in batch to reduce peak memory usage in extreme cases
    std::size_t max_batch_size = min(std::size_t(100000), tree->sparsetree.size());
    rmm::device_uvector<NodeT> d_tree(max_batch_size, builder_stream);
    rmm::device_uvector<InstanceRange> d_instance_ranges(max_batch_size, builder_stream);
    rmm::device_uvector<DataT> d_leaves(max_batch_size * dataset.num_outputs, builder_stream);

    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf, params.split_criterion);
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
      size_t smem_size = sizeof(BinT) * dataset.num_outputs;
      launchLeafKernel(objective,
                       dataset,
                       d_tree.data(),
                       d_instance_ranges.data(),
                       d_leaves.data(),
                       batch_size,
                       smem_size,
                       builder_stream);
      raft::update_host(tree->vector_leaf.data() + batch_begin * dataset.num_outputs,
                        d_leaves.data(),
                        batch_size * dataset.num_outputs,
                        builder_stream);
    }
  }
};  // end Builder

}  // namespace DT
}  // namespace ML
