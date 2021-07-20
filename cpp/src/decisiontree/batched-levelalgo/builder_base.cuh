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

  WorkloadInfo<IdxT>* workload_info;
  WorkloadInfo<IdxT>* h_workload_info;

  int max_blocks = 0;

  /** host copy of the number of new nodes in current branch */
  IdxT* h_n_nodes;
  /** host copy of the number of leaves created so far */
  IdxT* h_n_leaves;
  /** total number of nodes created so far */
  IdxT h_total_nodes;
  /** range of the currently worked upon nodes */
  IdxT node_start, node_end;
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
    size_t d_wsize, h_wsize;
    ML::PUSH_RANGE("Builder::workspaceSize @builder_base.cuh [batched-levelalgo]");
    auto max_batch   = params.max_batch_size;
    size_t nHistBins = max_batch * (params.n_bins) * n_blks_for_cols * input.numOutputs;

    d_wsize = 0;
    d_wsize += calculateAlignedBytes(sizeof(IdxT));                               // n_nodes
    d_wsize += calculateAlignedBytes(sizeof(BinT) * nHistBins);                   // hist
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch * n_blks_for_cols);  // done_count
    d_wsize += calculateAlignedBytes(sizeof(int) * max_batch);                    // mutex
    d_wsize += calculateAlignedBytes(sizeof(IdxT));                               // n_leaves
    d_wsize += calculateAlignedBytes(sizeof(IdxT));                               // n_depth
    d_wsize += calculateAlignedBytes(sizeof(SplitT) * max_batch);                 // splits
    d_wsize += calculateAlignedBytes(sizeof(NodeT) * max_batch);                  // curr_nodes
    d_wsize += calculateAlignedBytes(sizeof(NodeT) * 2 * max_batch);              // next_nodes
    d_wsize +=                                                                    // workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);

    // all nodes in the tree
    h_wsize = calculateAlignedBytes(sizeof(IdxT));   // h_n_nodes
    h_wsize += calculateAlignedBytes(sizeof(IdxT));  // h_n_leaves
    h_wsize +=                                       // h_workload_info
      calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);

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
    n_leaves = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT));
    n_depth = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(IdxT));
    splits = reinterpret_cast<SplitT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(SplitT) * max_batch);
    curr_nodes = reinterpret_cast<NodeT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(NodeT) * max_batch);
    next_nodes = reinterpret_cast<NodeT*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(NodeT) * 2 * max_batch);
    workload_info = reinterpret_cast<WorkloadInfo<IdxT>*>(d_wspace);
    d_wspace += calculateAlignedBytes(sizeof(WorkloadInfo<IdxT>) * max_blocks);
    // host
    h_n_nodes = reinterpret_cast<IdxT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(IdxT));
    h_n_leaves = reinterpret_cast<IdxT*>(h_wspace);
    h_wspace += calculateAlignedBytes(sizeof(IdxT));
    h_workload_info = reinterpret_cast<WorkloadInfo<IdxT>*>(h_wspace);
    ML::POP_RANGE();
  }

  /**
   * @brief Main training method. To be called only after `assignWorkspace()`
   *
   * @param[out] h_nodes    list of nodes (must be allocated using
   *                        cudaMallocHost!)
   * @param[out] num_leaves number of leaves created in the tree
   * @param[out] depth      max depth of the built tree
   * @param[in]  s          cuda steam
   */
  void train(std::vector<Node<DataT, LabelT, IdxT>>& h_nodes,
             IdxT& num_leaves,
             IdxT& depth,
             cudaStream_t s)
  {
    ML::PUSH_RANGE("Builder::train @builder_base.cuh [batched-levelalgo]");
    init(h_nodes, s);
    while (node_start < node_end) {
      IdxT new_nodes = doSplit(h_nodes, s);
      h_total_nodes += new_nodes;
      updateNodeRange();
    }
    raft::update_host(&num_leaves, n_leaves, 1, s);
    raft::update_host(&depth, n_depth, 1, s);
    ML::POP_RANGE();
  }

  size_t nodeSplitSmemSize()
  {
    return std::max(2 * sizeof(IdxT) * TPB_SPLIT, sizeof(BinT) * input.numOutputs);
  }

 private:
  /**
   * @brief Initialize buffers and state
   *
   * @param[out] h_nodes list of nodes (must be allocated using cudaMallocHost!)
   * @param[in]  s       cuda stream
   */
  void init(std::vector<Node<DataT, LabelT, IdxT>>& h_nodes, cudaStream_t s)
  {
    *h_n_nodes      = 0;
    auto max_batch  = params.max_batch_size;
    auto n_col_blks = n_blks_for_cols;
    CUDA_CHECK(cudaMemsetAsync(done_count, 0, sizeof(int) * max_batch * n_col_blks, s));
    CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, s));
    CUDA_CHECK(cudaMemsetAsync(n_leaves, 0, sizeof(IdxT), s));
    CUDA_CHECK(cudaMemsetAsync(n_depth, 0, sizeof(IdxT), s));
    node_start = 0;
    node_end = h_total_nodes = 1;  // start with root node
    h_nodes.resize(1);
    h_nodes[0].initSpNode();
    h_nodes[0].start          = 0;
    h_nodes[0].count          = input.nSampledRows;
    h_nodes[0].depth          = 0;
    h_nodes[0].info.unique_id = 0;
  }

  /** check whether any more nodes need to be processed or not */
  bool isOver() const { return node_end == h_total_nodes; }

  /**
   * @brief After the current batch is finished processing, update the range
   *        of nodes to be worked upon in the next batch
   */
  void updateNodeRange()
  {
    node_start           = node_end;
    auto nodes_remaining = h_total_nodes - node_end;
    node_end             = std::min(nodes_remaining, params.max_batch_size) + node_end;
  }

  auto updateWorkloadInfo(std::vector<Node<DataT, LabelT, IdxT>>& h_nodes, cudaStream_t s)
  {
    auto batchSize                     = node_end - node_start;
    size_t total_samples_in_curr_batch = 0;
    size_t n_large_nodes_in_curr_batch =
      0;  // large nodes are nodes having training instances larger than block size, hence require
          // global memory for histogram construction
    size_t total_num_blocks = 0;
    for (int n = 0; n < batchSize; n++) {
      total_samples_in_curr_batch += h_nodes[node_start + n].count;
      size_t num_blocks = raft::ceildiv(h_nodes[node_start + n].count, TPB_DEFAULT);
      num_blocks        = std::max(size_t(1), num_blocks);

      if (num_blocks > 1) ++n_large_nodes_in_curr_batch;

      bool is_leaf = leafBasedOnParams<DataT, IdxT>(h_nodes[node_start + n].depth,
                                                    params.max_depth,
                                                    params.min_samples_split,
                                                    params.max_leaves,
                                                    h_n_leaves,
                                                    h_nodes[node_start + n].count);
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
  IdxT doSplit(std::vector<Node<DataT, LabelT, IdxT>>& h_nodes, cudaStream_t s)
  {
    ML::PUSH_RANGE("Builder::doSplit @bulder_base.cuh [batched-levelalgo]");
    auto batchSize = node_end - node_start;
    // start fresh on the number of *new* nodes created in this batch
    CUDA_CHECK(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), s));
    initSplit<DataT, IdxT, TPB_DEFAULT>(splits, batchSize, s);

    // get the current set of nodes to be worked upon
    raft::update_device(curr_nodes, h_nodes.data() + node_start, batchSize, s);
    CUDA_CHECK(
      cudaMemsetAsync(next_nodes, 0, sizeof(Node<DataT, LabelT, IdxT>) * 2 * batchSize, s));

    auto [total_blocks, large_blocks] = this->updateWorkloadInfo(h_nodes, s);

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
                                              next_nodes,
                                              n_nodes,
                                              splits,
                                              n_leaves,
                                              h_total_nodes,
                                              n_depth);
    CUDA_CHECK(cudaGetLastError());
    ML::POP_RANGE();
    // copy the updated (due to leaf creation) and newly created child nodes
    raft::update_host(h_n_nodes, n_nodes, 1, s);
    raft::update_host(h_n_leaves, n_leaves, 1, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    h_nodes.resize(h_nodes.size() + batchSize + *h_n_nodes);
    raft::update_host(h_nodes.data() + node_start, curr_nodes, batchSize, s);
    raft::update_host(h_nodes.data() + h_total_nodes, next_nodes, *h_n_nodes, s);
    ML::POP_RANGE();
    return *h_n_nodes;
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
