/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "builder_kernels.cuh"

#include <common/grid_sync.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <cstdint>
#include <cstdio>

namespace ML {
namespace DT {

static constexpr int TPB_DEFAULT = 128;

struct NodeSplitPartitionState {
  std::int64_t left_count;
  bool valid_row;
  bool goes_left;
};

struct NodeSplitPartitionScanOp {
  __host__ __device__ NodeSplitPartitionState operator()(const NodeSplitPartitionState& lhs,
                                                         const NodeSplitPartitionState& rhs) const
  {
    return {lhs.left_count + rhs.left_count, rhs.valid_row, rhs.goes_left};
  }
};

// Output side of the segmented partition scan. The scan supplies the
// inclusive left count and current row side for each logical row slot in its
// node segment; this writer uses that state to place the row into the temporary
// partition buffer.
template <typename DataT, typename LabelT, int TPB>
struct NodeSplitPartitionWriter {
  Dataset<DataT, LabelT> dataset;
  const NodeWorkItem* work_items;
  const Split<DataT>* splits;
  const WorkloadInfo* workload_info;
  std::int64_t* partition_row_ids;

  __host__ __device__ void operator()(std::ptrdiff_t index, NodeSplitPartitionState state) const
  {
    if (!state.valid_row) { return; }

    const auto slot              = std::size_t(index);
    const auto workload_info_cta = workload_info[slot / TPB];
    const auto nid               = workload_info_cta.nodeid;
    const auto work_item         = work_items[nid];
    const auto split             = splits[nid];

    const auto range_start = work_item.instances.begin;
    const auto range_pos   = std::size_t(workload_info_cta.offset_blockid) * TPB + slot % TPB;

    const auto row             = dataset.row_ids[range_start + range_pos];
    const auto rank            = state.goes_left ? state.left_count - std::int64_t(1)
                                                 : static_cast<std::int64_t>(range_pos) - state.left_count;
    const auto out_idx         = range_start + (state.goes_left ? rank : split.nLeft + rank);
    partition_row_ids[out_idx] = row;
  }
};

// Copy back only ranges for nodes that actually split. Leaf/invalid nodes keep
// their existing row-id order because the scan writer skips them too.
template <typename DataT, typename LabelT, int TPB>
static __global__ void nodeSplitCopyBackKernel(const std::int64_t min_samples_leaf,
                                               const DataT min_impurity_decrease,
                                               const Dataset<DataT, LabelT> dataset,
                                               const NodeWorkItem* work_items,
                                               const Split<DataT>* splits,
                                               const WorkloadInfo* workload_info,
                                               const std::int64_t* partition_row_ids)
{
  const auto workload_info_cta = workload_info[blockIdx.x];
  const auto nid               = workload_info_cta.nodeid;
  const auto work_item         = work_items[nid];
  const auto split             = splits[nid];
  if (SplitNotValid(split, min_impurity_decrease, min_samples_leaf, work_item.instances.count)) {
    return;
  }

  const auto range_start = work_item.instances.begin;
  const auto range_len   = work_item.instances.count;
  const auto range_pos   = std::size_t(workload_info_cta.offset_blockid) * blockDim.x + threadIdx.x;
  if (range_pos < range_len) {
    const auto idx       = range_start + range_pos;
    dataset.row_ids[idx] = partition_row_ids[idx];
  }
}

template <typename DataT, typename LabelT, int TPB>
void launchNodeSplitKernel(const std::int64_t min_samples_leaf,
                           const DataT min_impurity_decrease,
                           const Dataset<DataT, LabelT>& dataset,
                           const NodeWorkItem* work_items,
                           const Split<DataT>* splits,
                           const WorkloadInfo* workload_info,
                           size_t n_blocks_dimx,
                           std::int64_t* partition_row_ids,
                           cudaStream_t builder_stream)
{
  if (n_blocks_dimx == 0) return;

  // Each slot corresponds to one thread lane in the tiled workload_info layout.
  // workload_info is grouped by node, so scan-by-key resets ranks at node boundaries.
  const auto n_slots = n_blocks_dimx * TPB;
  auto exec_policy   = rmm::exec_policy(builder_stream);
  auto slots_begin   = thrust::make_counting_iterator<std::size_t>(0);

  auto node_key = [workload_info] __host__ __device__(std::size_t slot) {
    return workload_info[slot / TPB].nodeid;
  };
  auto partition_state = [=] __host__ __device__(std::size_t slot) {
    const auto workload_info_cta = workload_info[slot / TPB];
    const auto nid               = workload_info_cta.nodeid;
    const auto work_item         = work_items[nid];
    const auto split             = splits[nid];
    if (SplitNotValid(split, min_impurity_decrease, min_samples_leaf, work_item.instances.count)) {
      return NodeSplitPartitionState{std::int64_t(0), false, false};
    }

    const auto range_pos = std::size_t(workload_info_cta.offset_blockid) * TPB + slot % TPB;
    if (range_pos >= work_item.instances.count) {
      return NodeSplitPartitionState{std::int64_t(0), false, false};
    }

    const auto row       = dataset.row_ids[work_item.instances.begin + range_pos];
    const auto col_idx   = std::size_t(split.colid) * dataset.M + row;
    const auto goes_left = dataset.data[col_idx] <= split.quesval;
    return NodeSplitPartitionState{goes_left ? std::int64_t(1) : std::int64_t(0), true, goes_left};
  };

  // The scan input is a stream of per-slot partition states keyed by node id.
  // The scan output is a tabulated writer, so partition_row_ids is populated
  // during the scan rather than by a second scatter kernel.
  auto node_keys        = thrust::make_transform_iterator(slots_begin, node_key);
  auto partition_states = thrust::make_transform_iterator(slots_begin, partition_state);
  auto partition_writer =
    thrust::make_tabulate_output_iterator(NodeSplitPartitionWriter<DataT, LabelT, TPB>{
      dataset, work_items, splits, workload_info, partition_row_ids});
  thrust::inclusive_scan_by_key(exec_policy,
                                node_keys,
                                node_keys + n_slots,
                                partition_states,
                                partition_writer,
                                thrust::equal_to<std::int64_t>{},
                                NodeSplitPartitionScanOp{});

  // The original row_ids buffer remains the source during the scan, so copy back after it finishes.
  nodeSplitCopyBackKernel<DataT, LabelT, TPB>
    <<<n_blocks_dimx, TPB, 0, builder_stream>>>(min_samples_leaf,
                                                min_impurity_decrease,
                                                dataset,
                                                work_items,
                                                splits,
                                                workload_info,
                                                partition_row_ids);
}

template <typename DatasetT, typename NodeT, typename ObjectiveT, typename DataT>
static __global__ void leafKernel(ObjectiveT objective,
                                  DatasetT dataset,
                                  const NodeT* tree,
                                  const InstanceRange* instance_ranges,
                                  DataT* leaves)
{
  using BinT = typename ObjectiveT::BinT;
  extern __shared__ char shared_memory[];
  auto histogram = reinterpret_cast<BinT*>(shared_memory);
  auto node_id   = blockIdx.x;
  auto& node     = tree[node_id];
  auto range     = instance_ranges[node_id];
  if (!node.IsLeaf()) return;
  auto tid = threadIdx.x;
  for (int i = tid; i < dataset.num_outputs; i += blockDim.x) {
    histogram[i] = BinT();
  }
  __syncthreads();
  for (auto i = range.begin + tid; i < range.begin + range.count; i += blockDim.x) {
    auto row   = dataset.row_ids[i];
    auto label = dataset.labels[row];
    objective.IncrementHistogram(histogram, 1, 0, label, dataset, row);
  }
  __syncthreads();
  if (tid == 0) {
    ObjectiveT::SetLeafVector(
      histogram, dataset.num_outputs, leaves + dataset.num_outputs * node_id);
  }
}

template <typename DatasetT, typename NodeT, typename ObjectiveT, typename DataT>
void launchLeafKernel(ObjectiveT objective,
                      DatasetT& dataset,
                      const NodeT* tree,
                      const InstanceRange* instance_ranges,
                      DataT* leaves,
                      int batch_size,
                      size_t smem_size,
                      cudaStream_t builder_stream)
{
  int num_blocks = batch_size;
  leafKernel<<<num_blocks, TPB_DEFAULT, smem_size, builder_stream>>>(
    objective, dataset, tree, instance_ranges, leaves);
}

/**
 * @brief For every threadblock, converts the smem pdf-histogram to
 *        cdf-histogram inplace using inclusive block-sum-scan and returns
 *        the total_sum
 * @return The total sum aggregated over the sumscan,
 *         as well as the modified cdf-histogram pointer
 */
template <typename BinT, int TPB>
DI BinT pdf_to_cdf(BinT* shared_histogram, int n_bins)
{
  // Blockscan instance preparation
  typedef cub::BlockScan<BinT, TPB> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // variable to accumulate aggregate of sumscans of previous iterations
  BinT total_aggregate = BinT();

  for (int tix = threadIdx.x; tix < raft::ceildiv(n_bins, TPB) * TPB; tix += blockDim.x) {
    BinT result;
    BinT block_aggregate;
    BinT element = tix < n_bins ? shared_histogram[tix] : BinT();
    BlockScan(temp_storage).InclusiveSum(element, result, block_aggregate);
    __syncthreads();
    if (tix < n_bins) { shared_histogram[tix] = result + total_aggregate; }
    total_aggregate += block_aggregate;
  }
  // return the total sum
  return total_aggregate;
}

template <typename DataT, typename LabelT, int TPB, typename ObjectiveT>
static __global__ void computeSplitKernel(typename ObjectiveT::BinT* histograms,
                                          int max_n_bins,
                                          std::int64_t min_samples_split,
                                          std::int64_t max_leaves,
                                          const Dataset<DataT, LabelT> dataset,
                                          const Quantiles<DataT> quantiles,
                                          const NodeWorkItem* work_items,
                                          std::int64_t colStart,
                                          const std::int64_t* column_samples,
                                          int* done_count,
                                          int* mutex,
                                          volatile Split<DataT>* splits,
                                          ObjectiveT objective,
                                          std::int64_t treeid,
                                          const WorkloadInfo* workload_info,
                                          uint64_t seed)
{
  using BinT = typename ObjectiveT::BinT;
  // dynamic shared memory
  extern __shared__ char smem[];
  constexpr int n_split_warps = (TPB + raft::WarpSize - 1) / raft::WarpSize;
  __shared__ __align__(alignof(
    Split<DataT>)) unsigned char split_scratch_storage[sizeof(Split<DataT>) * n_split_warps];
  auto* split_scratch = reinterpret_cast<Split<DataT>*>(split_scratch_storage);

  // Read workload info for this block
  WorkloadInfo workload_info_cta = workload_info[blockIdx.x];
  auto nid                       = workload_info_cta.nodeid;
  auto large_nid                 = workload_info_cta.large_nodeid;
  const auto work_item           = work_items[nid];
  auto range_start               = work_item.instances.begin;
  auto range_len                 = work_item.instances.count;

  auto offset_blockid = workload_info_cta.offset_blockid;
  auto num_blocks     = workload_info_cta.num_blocks;

  // obtaining the feature to test split on
  auto colIndex = colStart + blockIdx.y;
  auto col      = column_samples[nid * dataset.n_sampled_cols + colIndex];

  // getting the n_bins for that feature
  int n_bins = quantiles.n_bins_array[col];

  auto end                  = range_start + range_len;
  auto shared_histogram_len = n_bins * objective.NumClasses();
  auto* shared_histogram    = alignPointer<BinT>(smem);
  auto* shared_quantiles    = alignPointer<DataT>(shared_histogram + shared_histogram_len);
  auto* shared_done         = alignPointer<int>(shared_quantiles + n_bins);
  auto stride               = blockDim.x * num_blocks;
  auto tid                  = threadIdx.x + offset_blockid * blockDim.x;

  // populating shared memory with initial values
  for (int i = threadIdx.x; i < shared_histogram_len; i += blockDim.x)
    shared_histogram[i] = BinT();
  for (int b = threadIdx.x; b < n_bins; b += blockDim.x)
    shared_quantiles[b] = quantiles.quantiles_array[max_n_bins * col + b];

  // synchronizing above changes across block
  __syncthreads();

  // compute pdf shared histogram for all bins for all classes in shared mem

  // Must be 64 bit - can easily grow larger than a 32 bit int
  std::size_t col_offset = std::size_t(col) * dataset.M;
  for (auto i = range_start + tid; i < end; i += stride) {
    // each thread works over a data point and strides to the next
    auto row   = dataset.row_ids[i];
    auto data  = dataset.data[row + col_offset];
    auto label = dataset.labels[row];

    // Search bin indices so lower_bound uses 32-bit distance and advance arithmetic.
    auto bin_begin = cuda::counting_iterator<int, int>(0);
    auto bin_end   = bin_begin + n_bins;
    auto bin_it =
      ::cuda::std::lower_bound(bin_begin, bin_end, data, [shared_quantiles](int bin, DataT value) {
        return shared_quantiles[bin] < value;
      });
    int bin   = bin_it == bin_end ? n_bins - 1 : *bin_it;
    int start = static_cast<int>(bin);
    // ++shared_histogram[start]
    objective.IncrementHistogram(shared_histogram, n_bins, start, label, dataset, row);
  }

  // synchronizing above changes across block
  __syncthreads();
  if (num_blocks > 1) {
    // update the corresponding global location
    auto histograms_offset =
      ((large_nid * gridDim.y) + blockIdx.y) * max_n_bins * objective.NumClasses();
    for (int i = threadIdx.x; i < shared_histogram_len; i += blockDim.x) {
      BinT::AtomicAdd(histograms + histograms_offset + i, shared_histogram[i]);
    }

    __threadfence();  // for commit guarantee
    __syncthreads();

    // last threadblock will go ahead and compute the best split
    bool last = MLCommon::signalDone(
      done_count + nid * gridDim.y + blockIdx.y, num_blocks, offset_blockid == 0, shared_done);
    // if not the last threadblock, exit
    if (!last) return;

    // store the complete global histogram in shared memory of last block
    for (int i = threadIdx.x; i < shared_histogram_len; i += blockDim.x)
      shared_histogram[i] = histograms[histograms_offset + i];

    __syncthreads();
  }

  // PDF to CDF inplace in `shared_histogram`
  for (int c = 0; c < objective.NumClasses(); ++c) {
    // left to right scan operation for scanning
    // "lesser-than-or-equal" counts
    BinT total_sum = pdf_to_cdf<BinT, TPB>(shared_histogram + n_bins * c, n_bins);
    // now, `shared_histogram[n_bins * c + i]` will have count of datapoints of class `c`
    // that are less than or equal to `shared_quantiles[i]`.
  }

  __syncthreads();

  // calculate the best candidate bins (one for each thread in the block) in current feature and
  // corresponding information gain for splitting
  Split<DataT> sp = objective.Gain(shared_histogram, shared_quantiles, col, range_len, n_bins);

  __syncthreads();

  // calculate best bins among candidate bins per feature using warp reduce
  // then atomically update across features to get best split per node
  // (in split[nid])
  sp.evalBestSplit(split_scratch, splits + nid, mutex + nid, shared_quantiles, n_bins);
}

template <typename DataT, typename LabelT, int TPB, typename ObjectiveT>
void launchComputeSplitKernel(typename ObjectiveT::BinT* histograms,
                              int max_n_bins,
                              std::int64_t min_samples_split,
                              std::int64_t max_leaves,
                              const Dataset<DataT, LabelT>& dataset,
                              const Quantiles<DataT>& quantiles,
                              const NodeWorkItem* work_items,
                              std::int64_t colStart,
                              const std::int64_t* column_samples,
                              int* done_count,
                              int* mutex,
                              volatile Split<DataT>* splits,
                              ObjectiveT& objective,
                              std::int64_t treeid,
                              const WorkloadInfo* workload_info,
                              uint64_t seed,
                              dim3 grid,
                              size_t smem_size,
                              cudaStream_t builder_stream)
{
  computeSplitKernel<DataT, LabelT, TPB, ObjectiveT>
    <<<grid, TPB, smem_size, builder_stream>>>(histograms,
                                               max_n_bins,
                                               min_samples_split,
                                               max_leaves,
                                               dataset,
                                               quantiles,
                                               work_items,
                                               colStart,
                                               column_samples,
                                               done_count,
                                               mutex,
                                               splits,
                                               objective,
                                               treeid,
                                               workload_info,
                                               seed);
}

}  // namespace DT
}  // namespace ML
