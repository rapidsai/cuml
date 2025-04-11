/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "kernels/builder_kernels.cuh"

#include <common/Timer.h>

#include <cuml/common/pinned_host_vector.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <deque>
#include <memory>
#include <utility>

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
  IdxT* colids;
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
    d_wsize += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);  // colids

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
    colids = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT) * max_batch * dataset.n_sampled_cols);

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
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, work_items.size(), builder_stream);

    // get the current set of nodes to be worked upon
    raft::update_device(d_work_items, work_items.data(), work_items.size(), builder_stream);

    auto [n_blocks_dimx, n_large_nodes] = this->updateWorkloadInfo(work_items);

    // do feature-sampling
    if (dataset.n_sampled_cols != dataset.N) {
      raft::common::nvtx::range fun_scope("feature-sampling");
      constexpr int block_threads          = 128;
      constexpr int max_samples_per_thread = 72;  // register spillage if more than this limit
      // decide if the problem size is suitable for the excess-sampling strategy.
      //
      // our required shared memory is a function of number of samples we'll need to sample (in
      // parallel, with replacement) in excess to get 'k' uniques out of 'n' features. estimated
      // static shared memory required by cub's block-wide collectives:
      // max_samples_per_thread * block_threads * sizeof(IdxT)
      //
      // The maximum items to sample ( the constant `max_samples_per_thread` to be set at
      // compile-time) is calibrated so that:
      // 1. There is no register spills and accesses to global memory
      // 2. The required static shared memory (ie, `max_samples_per_thread * block_threads *
      // sizeof(IdxT)` does not exceed 46KB.
      //
      // number of samples we'll need to sample (in parallel, with replacement), to expect 'k'
      // unique samples from 'n' is given by the following equation: log(1 - k/n)/log(1 - 1/n) ref:
      // https://stats.stackexchange.com/questions/296005/the-expected-number-of-unique-elements-drawn-with-replacement
      IdxT n_parallel_samples =
        std::ceil(raft::log(1 - double(dataset.n_sampled_cols) / double(dataset.N)) /
                  (raft::log(1 - 1.f / double(dataset.N))));
      // maximum sampling work possible by all threads in a block :
      // `max_samples_per_thread * block_thread`
      // dynamically calculated sampling work to be done per block:
      // `n_parallel_samples`
      // former must be greater or equal to than latter for excess-sampling-based strategy
      if (max_samples_per_thread * block_threads >= n_parallel_samples) {
        raft::common::nvtx::range fun_scope("excess-sampling-based approach");
        dim3 grid;
        grid.x = work_items.size();
        grid.y = 1;
        grid.z = 1;

        if (n_parallel_samples <= block_threads)
          // each thread randomly samples only 1 sample
          excess_sample_with_replacement_kernel<IdxT, 1, block_threads>
            <<<grid, block_threads, 0, builder_stream>>>(colids,
                                                         d_work_items,
                                                         work_items.size(),
                                                         treeid,
                                                         seed,
                                                         dataset.N,
                                                         dataset.n_sampled_cols,
                                                         n_parallel_samples);
        else
          // each thread does more work and samples `max_samples_per_thread` samples
          excess_sample_with_replacement_kernel<IdxT, max_samples_per_thread, block_threads>
            <<<grid, block_threads, 0, builder_stream>>>(colids,
                                                         d_work_items,
                                                         work_items.size(),
                                                         treeid,
                                                         seed,
                                                         dataset.N,
                                                         dataset.n_sampled_cols,
                                                         n_parallel_samples);
        raft::common::nvtx::pop_range();
      } else {
        raft::common::nvtx::range fun_scope("reservoir-sampling-based approach");
        // using algo-L (reservoir sampling) strategy to sample 'dataset.n_sampled_cols' unique
        // features from 'dataset.N' total features
        dim3 grid;
        grid.x = (work_items.size() + 127) / 128;
        grid.y = 1;
        grid.z = 1;
        algo_L_sample_kernel<<<grid, block_threads, 0, builder_stream>>>(
          colids, d_work_items, work_items.size(), treeid, seed, dataset.N, dataset.n_sampled_cols);
        raft::common::nvtx::pop_range();
      }
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      raft::common::nvtx::pop_range();
    }

    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    for (IdxT c = 0; c < dataset.n_sampled_cols; c += n_blks_for_cols) {
      computeSplit(c, n_blocks_dimx, n_large_nodes);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    // create child nodes (or make the current ones leaf)
    raft::common::nvtx::push_range("nodeSplitKernel @builder.cuh [batched-levelalgo]");
    launchNodeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>(params.max_depth,
                                                            params.min_samples_leaf,
                                                            params.min_samples_split,
                                                            params.max_leaves,
                                                            params.min_impurity_decrease,
                                                            dataset,
                                                            d_work_items,
                                                            work_items.size(),
                                                            splits,
                                                            builder_stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::common::nvtx::pop_range();
    raft::update_host(h_splits, splits, work_items.size(), builder_stream);
    handle.sync_stream(builder_stream);
    return std::make_tuple(h_splits, work_items.size());
  }

  auto computeSplitSmemSize()
  {
    size_t smem_size_1 =
      params.max_n_bins * dataset.num_outputs * sizeof(BinT) +  // shared_histogram size
      params.max_n_bins * sizeof(DataT) +                       // shared_quantiles size
      sizeof(int);                                              // shared_done size
    // Extra room for alignment (see alignPointer in
    // computeSplitKernel)
    smem_size_1 += sizeof(DataT) + 3 * sizeof(int);
    // Calculate the shared memory needed for evalBestSplit
    size_t smem_size_2 = raft::ceildiv(TPB_DEFAULT, raft::WarpSize) * sizeof(SplitT);
    // Pick the max of two
    auto available_smem = handle.get_device_properties().sharedMemPerBlock;
    size_t smem_size    = std::max(smem_size_1, smem_size_2);
    ASSERT(available_smem >= smem_size, "Not enough shared memory. Consider reducing max_n_bins.");
    return smem_size;
  }

  void computeSplit(IdxT col, size_t n_blocks_dimx, size_t n_large_nodes)
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
    // classes, features and (large)nodes.
    int len_histograms = n_bins * n_classes * n_blocks_dimy * n_large_nodes;
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms, 0, sizeof(BinT) * len_histograms, builder_stream));
    // create the objective function object
    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf);
    // call the computeSplitKernel
    raft::common::nvtx::range kernel_scope("computeSplitKernel @builder.cuh [batched-levelalgo]");
    launchComputeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>(histograms,
                                                               params.max_n_bins,
                                                               params.max_depth,
                                                               params.min_samples_split,
                                                               params.max_leaves,
                                                               dataset,
                                                               quantiles,
                                                               d_work_items,
                                                               col,
                                                               colids,
                                                               done_count,
                                                               mutex,
                                                               splits,
                                                               objective,
                                                               treeid,
                                                               workload_info,
                                                               seed,
                                                               grid,
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
    rmm::device_uvector<DataT> d_leaves(max_batch_size * dataset.num_outputs, builder_stream);

    ObjectiveT objective(dataset.num_outputs, params.min_samples_leaf);
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
