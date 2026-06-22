/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels/builder_kernels.cuh"

#include <common/Timer.h>

#include <cuml/common/pinned_host_vector.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <type_traits>
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
  NodeQueue(DecisionTreeParams params,
            size_t max_nodes,
            size_t sampled_rows,
            int num_outputs,
            size_t global_sampled_rows = 0)
    : params(params), tree(std::make_shared<DT::TreeMetaDataNode<DataT, LabelT>>())
  {
    if (global_sampled_rows == 0) { global_sampled_rows = sampled_rows; }
    tree->num_outputs = num_outputs;
    tree->sparsetree.reserve(max_nodes);
    tree->sparsetree.emplace_back(NodeT::CreateLeafNode(global_sampled_rows));
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
    if (std::size_t(n.InstanceCount()) < std::size_t(params.min_samples_split)) return false;
    if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) return false;
    return true;
  }

  void Push(const std::vector<NodeWorkItem>& work_items, const Split<DataT>* h_splits)
  {
    // Update node queue based on splits
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto global_split        = h_splits[i];
      auto item                = work_items[i];
      auto parent_range        = node_instances_.at(item.idx);
      auto parent_global_count = std::size_t(tree->sparsetree.at(item.idx).InstanceCount());
      if (global_split.best_metric_val <= params.min_impurity_decrease) { continue; }

      if (params.max_leaves != -1 && tree->leaf_counter >= params.max_leaves) break;

      auto left_global_count  = std::size_t(global_split.global_nLeft);
      auto right_global_count = parent_global_count - left_global_count;
      auto left_local_count   = std::size_t(global_split.local_nLeft);
      auto right_local_count  = parent_range.count - left_local_count;

      tree->sparsetree.at(item.idx) = NodeT::CreateSplitNode(global_split.colid,
                                                             global_split.quesval,
                                                             global_split.best_metric_val,
                                                             int64_t(tree->sparsetree.size()),
                                                             parent_global_count);
      tree->leaf_counter++;

      tree->sparsetree.emplace_back(NodeT::CreateLeafNode(left_global_count));
      node_instances_.emplace_back(InstanceRange{parent_range.begin, left_local_count});
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, node_instances_.back()});
      }

      tree->sparsetree.emplace_back(NodeT::CreateLeafNode(right_global_count));
      node_instances_.emplace_back(
        InstanceRange{parent_range.begin + left_local_count, right_local_count});
      if (this->IsExpandable(tree->sparsetree.back(), item.depth + 1)) {
        work_items_.emplace_back(
          NodeWorkItem{tree->sparsetree.size() - 1, item.depth + 1, node_instances_.back()});
      }

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
  typedef Split<DataT> SplitT;
  typedef Dataset<DataT, LabelT, IdxT> DatasetT;
  typedef Quantiles<DataT, IdxT> QuantilesT;

  /** default threads per block for most kernels in here */
  static constexpr int TPB_DEFAULT = 128;
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
  /** buffer of segmented histograms*/
  BinT* histograms;
  /** mutex array used for atomically updating best split */
  int* mutex;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  NodeWorkItem* d_work_items;
  /** device AOS to map CTA blocks along dimx to nodes of a batch */
  WorkloadInfo* workload_info;
  /** host AOS to map CTA blocks along dimx to nodes of a batch */
  WorkloadInfo* h_workload_info;
  /** maximum CTA blocks along dimx */
  int max_blocks_dimx = 0;
  /** host array of splits */
  SplitT* h_splits;
  /** packed histogram buffer used by distributed all-reduce */
  void* packed_histograms;
  /** number of blocks used to parallelize column-wise computations */
  int n_blks_for_cols = 10;
  /** Memory alignment value */
  const size_t align_value = 512;
  IdxT* column_samples;
  /** rmm device workspace buffer */
  rmm::device_uvector<char> d_buff;
  /** pinned host buffer to store the trained nodes */
  ML::pinned_host_vector<char> h_buff;
  /** true when a communicator with more than one rank is available */
  bool distributed;
  /** global root sample count in distributed mode */
  std::size_t global_sampled_rows;

  Builder(const raft::handle_t& handle,
          cudaStream_t s,
          IdxT treeid,
          uint64_t seed,
          const DecisionTreeParams& p,
          const DataT* data,
          const LabelT* labels,
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
              n_rows,
              n_cols,
              int(row_ids->size()),
              max(1, IdxT(params.max_features * n_cols)),
              row_ids->data(),
              n_classes},
      quantiles(q),
      d_buff(0, builder_stream),
      distributed(raft::resource::comms_initialized(handle) && handle.get_comms().get_size() > 1),
      global_sampled_rows(row_ids->size())
  {
    if (distributed) {
      // Each rank stores only its local row_ids, but tree metadata and split
      // validity checks need the global bootstrap sample count.
      auto local_count = static_cast<std::uint64_t>(row_ids->size());
      rmm::device_uvector<std::uint64_t> count_buffer(1, builder_stream);
      raft::update_device(count_buffer.data(), &local_count, 1, builder_stream);
      handle.get_comms().allreduce(
        count_buffer.data(), count_buffer.data(), 1, raft::comms::op_t::SUM, builder_stream);
      ASSERT(handle.get_comms().sync_stream(builder_stream) == raft::comms::status_t::SUCCESS,
             "An error occurred in the distributed RF row-count all-reduce.");
      auto global_count = std::uint64_t{0};
      raft::update_host(&global_count, count_buffer.data(), 1, builder_stream);
      handle.sync_stream(builder_stream);
      global_sampled_rows = global_count;
    }
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
  template <typename T = BinT>
  size_t packedHistogramWorkspaceSize(size_t len) const
  {
    size_t size = calculateAlignedBytes(sizeof(std::uint64_t) * len);
    if constexpr (has_label_sum_v<T>) {
      size += calculateAlignedBytes(sizeof(double) * len);
    }
    if constexpr (has_weight_v<T>) {
      size += calculateAlignedBytes(sizeof(double) * len);
    }
    return size;
  }

  auto workspaceSize() const
  {
    size_t d_wsize = 0, h_wsize = 0;
    raft::common::nvtx::range fun_scope("Builder::workspaceSize @builder.cuh [batched-levelalgo]");
    auto max_batch = params.max_batch_size;
    size_t max_len_histograms =
      max_batch * params.max_n_bins * n_blks_for_cols * dataset.num_outputs;

    d_wsize += calculateAlignedBytes(sizeof(BinT) * max_len_histograms);  // histograms
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch);            // mutex
    d_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);         // splits
    d_wsize += calculateAlignedBytes(sizeof(NodeWorkItem) * max_batch);   // d_work_Items
    d_wsize +=                                                            // workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo) * max_blocks_dimx);
    d_wsize += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);  // column_samples

    // all nodes in the tree
    h_wsize +=  // h_workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo) * max_blocks_dimx);
    h_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);  // splits
    d_wsize += packedHistogramWorkspaceSize(max_len_histograms);

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
    auto max_batch = params.max_batch_size;
    size_t max_len_histograms =
      max_batch * (params.max_n_bins) * n_blks_for_cols * dataset.num_outputs;
    // device
    histograms = reinterpret_cast<BinT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(BinT) * max_len_histograms);
    mutex = reinterpret_cast<int*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(int) * max_batch);
    splits = reinterpret_cast<SplitT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    d_work_items = reinterpret_cast<NodeWorkItem*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(NodeWorkItem) * max_batch);
    workload_info = reinterpret_cast<WorkloadInfo*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(WorkloadInfo) * max_blocks_dimx);
    column_samples = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);

    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, builder_stream));

    // host
    h_workload_info = reinterpret_cast<WorkloadInfo*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(WorkloadInfo) * max_blocks_dimx);
    h_splits = reinterpret_cast<SplitT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    packed_histograms = reinterpret_cast<void*>(d_wspace);
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
      params, this->maxNodes(), dataset.n_sampled_rows, dataset.num_outputs, global_sampled_rows);
    while (queue.HasWork()) {
      auto work_items = queue.Pop();
      queue.Push(work_items, doSplit(work_items));
    }
    auto tree = queue.GetTree();
    this->SetLeafPredictions(tree, queue.GetInstanceRanges());
    tree->train_time = timer.getElapsedMilliseconds();
    return tree;
  }

 private:
  auto updateWorkloadInfo(const std::vector<NodeWorkItem>& work_items)
  {
    int n_blocks_dimx = 0;  // gridDim.x required for split histogram construction
    for (std::size_t i = 0; i < work_items.size(); i++) {
      auto item             = work_items[i];
      const auto node_id    = static_cast<int>(i);
      int n_blocks_per_node = static_cast<int>(
        std::max(raft::ceildiv(item.instances.count, size_t(TPB_DEFAULT)), size_t(1)));

      for (int b = 0; b < n_blocks_per_node; b++) {
        h_workload_info[n_blocks_dimx + b] = {node_id, b, n_blocks_per_node};
      }
      n_blocks_dimx += n_blocks_per_node;
    }
    raft::update_device(workload_info, h_workload_info, n_blocks_dimx, builder_stream);
    return n_blocks_dimx;
  }

  auto doSplit(const std::vector<NodeWorkItem>& work_items)
  {
    raft::common::nvtx::range fun_scope("Builder::doSplit @builder.cuh [batched-levelalgo]");
    const IdxT original_n_sampled_cols = dataset.n_sampled_cols;
    ASSERT(original_n_sampled_cols > 0 && original_n_sampled_cols <= dataset.N,
           "n_sampled_cols must be in [1, n_cols]");
    const std::size_t max_sampling_rounds =
      std::size_t((dataset.N + original_n_sampled_cols - 1) / original_n_sampled_cols);

    std::vector<SplitT> final_splits(work_items.size());
    std::vector<NodeWorkItem> active_items(work_items);
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
        final_splits[original_idx] = h_splits[i];
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

    RAFT_CUDA_TRY(cudaMemcpyAsync(splits,
                                  final_splits.data(),
                                  sizeof(SplitT) * work_items.size(),
                                  cudaMemcpyHostToDevice,
                                  builder_stream));
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);
    raft::common::nvtx::push_range("nodeSplitKernel @builder.cuh [batched-levelalgo]");
    launchNodeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>(params.min_impurity_decrease,
                                                            dataset,
                                                            d_work_items,
                                                            work_items.size(),
                                                            splits,
                                                            builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::common::nvtx::pop_range();
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    handle.sync_stream(builder_stream);
    return h_splits;
  }

  void computeBestSplits(const std::vector<NodeWorkItem>& work_items,
                         uint64_t sampling_seed,
                         IdxT sample_offset)
  {
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, work_items.size(), builder_stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(mutex, 0, sizeof(int) * params.max_batch_size, builder_stream));
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);
    auto n_blocks_dimx = this->updateWorkloadInfo(work_items);

    sampleFeatures(work_items, sampling_seed, sample_offset);

    for (IdxT c = 0; c < dataset.n_sampled_cols; c += n_blks_for_cols) {
      computeSplit(c, n_blocks_dimx, work_items.size());
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

  auto computeSplitSmemSize()
  {
    size_t smem_size_1 =
      params.max_n_bins * dataset.num_outputs * sizeof(BinT) +  // shared_histogram size
      params.max_n_bins * sizeof(DataT);                        // shared_quantiles size
    // Extra room for alignment (see alignPointer in the split kernels).
    smem_size_1 += sizeof(DataT);
    // Calculate the shared memory needed for evalBestSplit
    size_t smem_size_2 = raft::ceildiv(TPB_DEFAULT, raft::WarpSize) * sizeof(SplitT);
    // Pick the max of two
    auto available_smem = handle.get_device_properties().sharedMemPerBlock;
    size_t smem_size    = std::max(smem_size_1, smem_size_2);
    ASSERT(available_smem >= smem_size, "Not enough shared memory. Consider reducing max_n_bins.");
    return smem_size;
  }

  void allReduceHistograms(BinT* histograms_to_reduce, std::size_t len_histograms)
  {
    auto const& comm = handle.get_comms();
    auto* packed_base = reinterpret_cast<char*>(packed_histograms);
    double* packed_label_sums = nullptr;
    if constexpr (has_label_sum_v<BinT>) {
      packed_label_sums = reinterpret_cast<double*>(packed_base);
      packed_base += calculateAlignedBytes(sizeof(double) * len_histograms);
    }
    auto* packed_counts = reinterpret_cast<std::uint64_t*>(packed_base);
    packed_base += calculateAlignedBytes(sizeof(std::uint64_t) * len_histograms);
    double* packed_weights = nullptr;
    if constexpr (has_weight_v<BinT>) {
      packed_weights = reinterpret_cast<double*>(packed_base);
    }

    packHistograms(histograms_to_reduce,
                   packed_label_sums,
                   packed_counts,
                   packed_weights,
                   len_histograms,
                   builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    if constexpr (has_label_sum_v<BinT>) {
      comm.allreduce(packed_label_sums,
                     packed_label_sums,
                     len_histograms,
                     raft::comms::op_t::SUM,
                     builder_stream);
      ASSERT(comm.sync_stream(builder_stream) == raft::comms::status_t::SUCCESS,
             "An error occurred in the distributed RF label-sum histogram all-reduce.");
    }
    comm.allreduce(
      packed_counts, packed_counts, len_histograms, raft::comms::op_t::SUM, builder_stream);
    ASSERT(comm.sync_stream(builder_stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed RF count histogram all-reduce.");
    if constexpr (has_weight_v<BinT>) {
      comm.allreduce(packed_weights,
                     packed_weights,
                     len_histograms,
                     raft::comms::op_t::SUM,
                     builder_stream);
      ASSERT(comm.sync_stream(builder_stream) == raft::comms::status_t::SUCCESS,
             "An error occurred in the distributed RF weight histogram all-reduce.");
    }
    unpackHistograms(packed_label_sums,
                     packed_counts,
                     packed_weights,
                     histograms_to_reduce,
                     len_histograms,
                     builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  void computeSplit(IdxT col, size_t n_blocks_dimx, size_t work_items_size)
  {
    // if no instances to split, return
    if (n_blocks_dimx == 0) return;
    raft::common::nvtx::range fun_scope("Builder::computeSplit @builder.cuh [batched-levelalgo]");
    auto n_bins    = params.max_n_bins;
    auto n_classes = dataset.num_outputs;
    // if columns left to be processed lesser than `n_blks_for_cols`, shrink the blocks along dimy
    auto n_blocks_dimy = std::min(n_blks_for_cols, dataset.n_sampled_cols - col);
    // compute required dynamic shared memory
    auto smem_size = computeSplitSmemSize();
    dim3 grid(n_blocks_dimx, n_blocks_dimy, 1);
    // required total length (in bins) of the global segmented histograms over all
    // classes, features and nodes.
    int len_histograms = n_bins * n_classes * n_blocks_dimy * work_items_size;
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms, 0, sizeof(BinT) * len_histograms, builder_stream));
    raft::common::nvtx::range kernel_scope("split kernels @builder.cuh [batched-levelalgo]");
    launchComputeSplitHistogramKernel<DataT, LabelT, IdxT, TPB_DEFAULT>(histograms,
                                                                        params.max_n_bins,
                                                                        dataset,
                                                                        quantiles,
                                                                        d_work_items,
                                                                        col,
                                                                        column_samples,
                                                                        workload_info,
                                                                        grid,
                                                                        smem_size,
                                                                        builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    if (distributed) { allReduceHistograms(histograms, len_histograms); }
    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf, params.split_criterion);
    dim3 eval_grid(work_items_size, n_blocks_dimy, 1);
    launchEvaluateSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>(histograms,
                                                                params.max_n_bins,
                                                                dataset,
                                                                quantiles,
                                                                col,
                                                                column_samples,
                                                                mutex,
                                                                splits,
                                                                objective,
                                                                eval_grid,
                                                                smem_size,
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
    rmm::device_uvector<BinT> d_leaf_histograms(max_batch_size * dataset.num_outputs,
                                                builder_stream);
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

      RAFT_CUDA_TRY(cudaMemsetAsync(
        d_leaf_histograms.data(), 0, sizeof(BinT) * d_leaf_histograms.size(), builder_stream));
      RAFT_CUDA_TRY(
        cudaMemsetAsync(d_leaves.data(), 0, sizeof(DataT) * d_leaves.size(), builder_stream));
      size_t smem_size = sizeof(BinT) * dataset.num_outputs;
      launchLeafHistogramKernel(objective,
                                dataset,
                                d_tree.data(),
                                d_instance_ranges.data(),
                                d_leaf_histograms.data(),
                                batch_size,
                                smem_size,
                                builder_stream);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      if (distributed) {
        allReduceHistograms(d_leaf_histograms.data(), batch_size * dataset.num_outputs);
      }
      launchFinalizeLeafKernel(objective,
                               d_tree.data(),
                               d_leaf_histograms.data(),
                               d_leaves.data(),
                               batch_size,
                               dataset.num_outputs,
                               builder_stream);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      raft::update_host(tree->vector_leaf.data() + batch_begin * dataset.num_outputs,
                        d_leaves.data(),
                        batch_size * dataset.num_outputs,
                        builder_stream);
    }
  }
};  // end Builder

}  // namespace DT
}  // namespace ML
