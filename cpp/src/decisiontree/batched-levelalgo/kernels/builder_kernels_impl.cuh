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
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <cstdio>

namespace ML {
namespace DT {

static constexpr int TPB_DEFAULT = 128;

template <typename IdxT>
struct NodeSplitPartitionState {
  IdxT left_count;
  bool valid_row;
  bool goes_left;
};

template <typename IdxT>
struct NodeSplitPartitionScanOp {
  __host__ __device__ NodeSplitPartitionState<IdxT> operator()(
    const NodeSplitPartitionState<IdxT>& lhs, const NodeSplitPartitionState<IdxT>& rhs) const
  {
    return {lhs.left_count + rhs.left_count, rhs.valid_row, rhs.goes_left};
  }
};

// Output side of the segmented partition scan. The scan supplies the
// inclusive left count and current row side for each logical row slot in its
// node segment; this writer uses that state to place the row into the temporary
// partition buffer.
template <typename DataT, typename LabelT, typename IdxT, int TPB>
struct NodeSplitPartitionWriter {
  Dataset<DataT, LabelT, IdxT> dataset;
  const NodeWorkItem* work_items;
  const Split<DataT, IdxT>* splits;
  const WorkloadInfo<IdxT>* workload_info;
  IdxT* partition_row_ids;

  __host__ __device__ void operator()(std::ptrdiff_t index,
                                      NodeSplitPartitionState<IdxT> state) const
  {
    if (!state.valid_row) { return; }

    const auto slot              = std::size_t(index);
    const auto workload_info_cta = workload_info[slot / TPB];
    const auto nid               = workload_info_cta.nodeid;
    const auto work_item         = work_items[nid];
    const auto split             = splits[nid];

    const auto range_start = work_item.instances.begin;
    const auto range_pos   = std::size_t(workload_info_cta.offset_blockid) * TPB + slot % TPB;

    const auto row = dataset.row_ids[range_start + range_pos];
    const auto rank =
      state.goes_left ? state.left_count - IdxT(1) : IdxT(range_pos) - state.left_count;
    const auto out_idx         = range_start + (state.goes_left ? rank : split.nLeft + rank);
    partition_row_ids[out_idx] = row;
  }
};

// Copy back only ranges for nodes that actually split. Leaf/invalid nodes keep
// their existing row-id order because the scan writer skips them too.
template <typename DataT, typename LabelT, typename IdxT, int TPB>
static __global__ void nodeSplitCopyBackKernel(const IdxT min_samples_leaf,
                                               const DataT min_impurity_decrease,
                                               const Dataset<DataT, LabelT, IdxT> dataset,
                                               const NodeWorkItem* work_items,
                                               const Split<DataT, IdxT>* splits,
                                               const WorkloadInfo<IdxT>* workload_info,
                                               const IdxT* partition_row_ids)
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

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launchNodeSplitKernel(const IdxT min_samples_leaf,
                           const DataT min_impurity_decrease,
                           const Dataset<DataT, LabelT, IdxT>& dataset,
                           const NodeWorkItem* work_items,
                           const Split<DataT, IdxT>* splits,
                           const WorkloadInfo<IdxT>* workload_info,
                           size_t n_blocks_dimx,
                           IdxT* partition_row_ids,
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
      return NodeSplitPartitionState<IdxT>{IdxT(0), false, false};
    }

    const auto range_pos = std::size_t(workload_info_cta.offset_blockid) * TPB + slot % TPB;
    if (range_pos >= work_item.instances.count) {
      return NodeSplitPartitionState<IdxT>{IdxT(0), false, false};
    }

    const auto row       = dataset.row_ids[work_item.instances.begin + range_pos];
    const auto col_idx   = std::size_t(split.colid) * dataset.M + row;
    const auto goes_left = dataset.data[col_idx] <= split.quesval;
    return NodeSplitPartitionState<IdxT>{goes_left ? IdxT(1) : IdxT(0), true, goes_left};
  };

  // The scan input is a stream of per-slot partition states keyed by node id.
  // The scan output is a tabulated writer, so partition_row_ids is populated
  // during the scan rather than by a second scatter kernel.
  auto node_keys        = thrust::make_transform_iterator(slots_begin, node_key);
  auto partition_states = thrust::make_transform_iterator(slots_begin, partition_state);
  auto partition_writer =
    thrust::make_tabulate_output_iterator(NodeSplitPartitionWriter<DataT, LabelT, IdxT, TPB>{
      dataset, work_items, splits, workload_info, partition_row_ids});
  thrust::inclusive_scan_by_key(exec_policy,
                                node_keys,
                                node_keys + n_slots,
                                partition_states,
                                partition_writer,
                                thrust::equal_to<IdxT>{},
                                NodeSplitPartitionScanOp<IdxT>{});

  // The original row_ids buffer remains the source during the scan, so copy back after it finishes.
  nodeSplitCopyBackKernel<DataT, LabelT, IdxT, TPB>
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
 * @brief For every threadblock, converts a pdf-histogram to a
 *        cdf-histogram inplace using inclusive block-sum-scan and returns
 *        the total_sum
 * @return The total sum aggregated over the sumscan,
 *         as well as the modified cdf-histogram pointer
 */
template <typename BinT, typename IdxT, int TPB>
DI BinT pdf_to_cdf(BinT* histogram, IdxT n_bins)
{
  // Blockscan instance preparation
  typedef cub::BlockScan<BinT, TPB> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // variable to accumulate aggregate of sumscans of previous iterations
  BinT total_aggregate = BinT();

  for (IdxT tix = threadIdx.x; tix < raft::ceildiv(n_bins, TPB) * TPB; tix += blockDim.x) {
    BinT result;
    BinT block_aggregate;
    BinT element = tix < n_bins ? histogram[tix] : BinT();
    BlockScan(temp_storage).InclusiveSum(element, result, block_aggregate);
    __syncthreads();
    if (tix < n_bins) { histogram[tix] = result + total_aggregate; }
    total_aggregate += block_aggregate;
  }
  // return the total sum
  return total_aggregate;
}

template <typename DataT, typename LabelT, typename IdxT, int TPB, typename ObjectiveT>
static __global__ void computeSplitKernel(typename ObjectiveT::BinT* histograms,
                                          IdxT max_n_bins,
                                          IdxT min_samples_split,
                                          IdxT max_leaves,
                                          const Dataset<DataT, LabelT, IdxT> dataset,
                                          const Quantiles<DataT, IdxT> quantiles,
                                          const NodeWorkItem* work_items,
                                          IdxT colStart,
                                          const IdxT* column_samples,
                                          int* done_count,
                                          int* mutex,
                                          volatile Split<DataT, IdxT>* splits,
                                          ObjectiveT objective,
                                          IdxT treeid,
                                          const WorkloadInfo<IdxT>* workload_info,
                                          uint64_t seed,
                                          bool use_global_memory_histogram)
{
  using BinT = typename ObjectiveT::BinT;
  // dynamic shared memory
  extern __shared__ char smem[];
  constexpr int n_split_warps = (TPB + raft::WarpSize - 1) / raft::WarpSize;
  __shared__ __align__(alignof(Split<DataT, IdxT>)) unsigned char
    split_scratch_storage[sizeof(Split<DataT, IdxT>) * n_split_warps];
  auto* split_scratch = reinterpret_cast<Split<DataT, IdxT>*>(split_scratch_storage);

  // Read workload info for this block
  WorkloadInfo<IdxT> workload_info_cta = workload_info[blockIdx.x];
  IdxT nid                             = workload_info_cta.nodeid;
  IdxT large_nid                       = workload_info_cta.large_nodeid;
  const auto work_item                 = work_items[nid];
  auto range_start                     = work_item.instances.begin;
  auto range_len                       = work_item.instances.count;

  IdxT offset_blockid = workload_info_cta.offset_blockid;
  IdxT num_blocks     = workload_info_cta.num_blocks;

  // obtaining the feature to test split on
  IdxT colIndex = colStart + blockIdx.y;
  IdxT col      = column_samples[nid * dataset.n_sampled_cols + colIndex];

  // getting the n_bins for that feature
  int n_bins = quantiles.n_bins_array[col];

  auto n_classes            = objective.NumClasses();
  auto end                  = range_start + range_len;
  auto histogram_len        = n_bins * n_classes;
  auto* histogram           = static_cast<BinT*>(nullptr);
  auto* shared_done         = static_cast<int*>(nullptr);
  auto* quantiles_for_split = quantiles.quantiles_array + std::size_t(max_n_bins) * col;
  IdxT stride               = blockDim.x * num_blocks;
  IdxT tid                  = threadIdx.x + offset_blockid * blockDim.x;

  if (use_global_memory_histogram) {
    auto histograms_offset = (std::size_t(nid) * gridDim.y + blockIdx.y) * max_n_bins * n_classes;
    histogram              = histograms + histograms_offset;
    shared_done            = alignPointer<int>(smem);
  } else {
    histogram              = alignPointer<BinT>(smem);
    auto* shared_quantiles = alignPointer<DataT>(histogram + histogram_len);
    shared_done            = alignPointer<int>(shared_quantiles + n_bins);
    quantiles_for_split    = shared_quantiles;
    for (IdxT i = threadIdx.x; i < histogram_len; i += blockDim.x) {
      histogram[i] = BinT();
    }
    for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x) {
      shared_quantiles[b] = quantiles.quantiles_array[max_n_bins * col + b];
    }
  }

  __syncthreads();

  // compute pdf histogram for all bins for all classes

  // Must be 64 bit - can easily grow larger than a 32 bit int
  std::size_t col_offset = std::size_t(col) * dataset.M;
  for (auto i = range_start + tid; i < end; i += stride) {
    // each thread works over a data point and strides to the next
    auto row   = dataset.row_ids[i];
    auto data  = dataset.data[row + col_offset];
    auto label = dataset.labels[row];

    // `start` is lowest index such that data <= quantiles_for_split[start]
    IdxT start = lower_bound(quantiles_for_split, n_bins, data);
    // ++histogram[start]
    objective.IncrementHistogram(histogram, n_bins, start, label, dataset, row);
  }

  __syncthreads();
  if (use_global_memory_histogram) {
    __threadfence();  // for commit guarantee before the last block scores the split
    __syncthreads();

    bool last = MLCommon::signalDone(
      done_count + nid * gridDim.y + blockIdx.y, num_blocks, offset_blockid == 0, shared_done);
    if (!last) return;
  } else if (num_blocks > 1) {
    // Shared-memory histogram path: each block built a partial histogram, so unify those
    // partial histograms in global memory before scoring the split.
    // update the corresponding global location
    auto histograms_offset =
      (std::size_t(large_nid) * gridDim.y + blockIdx.y) * max_n_bins * n_classes;
    for (IdxT i = threadIdx.x; i < histogram_len; i += blockDim.x) {
      BinT::AtomicAdd(histograms + histograms_offset + i, histogram[i]);
    }

    __threadfence();  // for commit guarantee
    __syncthreads();

    // last threadblock will go ahead and compute the best split
    bool last = MLCommon::signalDone(
      done_count + nid * gridDim.y + blockIdx.y, num_blocks, offset_blockid == 0, shared_done);
    // if not the last threadblock, exit
    if (!last) return;

    // store the complete global histogram in shared memory of last block
    for (IdxT i = threadIdx.x; i < histogram_len; i += blockDim.x) {
      histogram[i] = histograms[histograms_offset + i];
    }

    __syncthreads();
  }

  // PDF to CDF inplace in `histogram`
  for (IdxT c = 0; c < n_classes; ++c) {
    // left to right scan operation for scanning
    // "lesser-than-or-equal" counts
    BinT total_sum = pdf_to_cdf<BinT, IdxT, TPB>(histogram + n_bins * c, n_bins);
    // now, `histogram[n_bins * c + i]` will have count of datapoints of class `c`
    // that are less than or equal to `quantiles_for_split[i]`.
  }

  __syncthreads();

  // calculate the best candidate bins (one for each thread in the block) in current feature and
  // corresponding information gain for splitting
  Split<DataT, IdxT> sp = objective.Gain(histogram, quantiles_for_split, col, range_len, n_bins);

  __syncthreads();

  // calculate best bins among candidate bins per feature using warp reduce
  // then atomically update across features to get best split per node
  // (in split[nid])
  sp.evalBestSplit(split_scratch, splits + nid, mutex + nid, quantiles_for_split, n_bins);
}

template <typename DataT, typename LabelT, typename IdxT, int TPB, typename ObjectiveT>
void launchComputeSplitKernel(typename ObjectiveT::BinT* histograms,
                              IdxT max_n_bins,
                              IdxT min_samples_split,
                              IdxT max_leaves,
                              const Dataset<DataT, LabelT, IdxT>& dataset,
                              const Quantiles<DataT, IdxT>& quantiles,
                              const NodeWorkItem* work_items,
                              IdxT colStart,
                              const IdxT* column_samples,
                              int* done_count,
                              int* mutex,
                              volatile Split<DataT, IdxT>* splits,
                              ObjectiveT& objective,
                              IdxT treeid,
                              const WorkloadInfo<IdxT>* workload_info,
                              uint64_t seed,
                              bool use_global_memory_histogram,
                              dim3 grid,
                              size_t smem_size,
                              cudaStream_t builder_stream)
{
  computeSplitKernel<DataT, LabelT, IdxT, TPB, ObjectiveT>
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
                                               seed,
                                               use_global_memory_histogram);
}

}  // namespace DT
}  // namespace ML
