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

#include <algorithm>
#include <cstdio>

namespace ML {
namespace DT {

static constexpr int TPB_DEFAULT = 128;

template <typename CountT>
struct NodeSplitPartitionState {
  CountT left_count;
  bool valid_row;
  bool goes_left;
};

template <typename CountT>
struct NodeSplitPartitionScanOp {
  __host__ __device__ NodeSplitPartitionState<CountT> operator()(
    const NodeSplitPartitionState<CountT>& lhs, const NodeSplitPartitionState<CountT>& rhs) const
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
  using CountT = typename Split<DataT>::CountT;

  Dataset<DataT, LabelT, IdxT> dataset;
  const NodeWorkItem* work_items;
  const Split<DataT>* splits;
  const WorkloadInfo* workload_info;
  IdxT* partition_row_ids;

  __host__ __device__ void operator()(std::ptrdiff_t index,
                                      NodeSplitPartitionState<CountT> state) const
  {
    if (!state.valid_row) { return; }

    const auto slot              = std::size_t(index);
    const auto workload_info_cta = workload_info[slot / TPB];
    const auto nid               = workload_info_cta.nodeid;
    const auto work_item         = work_items[nid];
    const auto split             = splits[nid];

    const auto range_start = work_item.instances.begin;
    const auto range_pos   = std::size_t(workload_info_cta.offset_blockid) * TPB + slot % TPB;

    const auto row              = dataset.row_ids[range_start + range_pos];
    const auto rank             = state.goes_left ? std::size_t(state.left_count - CountT{1})
                                                  : range_pos - std::size_t(state.left_count);
    const auto local_left_count = std::size_t(split.local_nLeft);
    const auto out_idx          = range_start + (state.goes_left ? rank : local_left_count + rank);
    partition_row_ids[out_idx]  = row;
  }
};

template <typename DataT, typename LabelT, typename IdxT, int TPB>
static __global__ void nodeSplitLocalCountKernel(const DataT min_impurity_decrease,
                                                 const Dataset<DataT, LabelT, IdxT> dataset,
                                                 const NodeWorkItem* work_items,
                                                 Split<DataT>* splits,
                                                 const WorkloadInfo* workload_info)
{
  using CountT = typename Split<DataT>::CountT;

  const auto workload_info_cta = workload_info[blockIdx.x];
  const auto nid               = workload_info_cta.nodeid;
  const auto work_item         = work_items[nid];
  const auto split             = splits[nid];
  if (SplitNotValid(split, min_impurity_decrease)) { return; }

  const auto range_pos = std::size_t(workload_info_cta.offset_blockid) * blockDim.x + threadIdx.x;
  if (range_pos >= work_item.instances.count) { return; }

  const auto row       = dataset.row_ids[work_item.instances.begin + range_pos];
  const auto col_idx   = std::size_t(split.colid) * dataset.M + row;
  const auto goes_left = dataset.data[col_idx] <= split.quesval;
  if (goes_left) { atomicAdd(&splits[nid].local_nLeft, CountT{1}); }
}

// Copy back only ranges for nodes that actually split. Leaf/invalid nodes keep
// their existing row-id order because the scan writer skips them too.
template <typename DataT, typename LabelT, typename IdxT, int TPB>
static __global__ void nodeSplitCopyBackKernel(const DataT min_impurity_decrease,
                                               const Dataset<DataT, LabelT, IdxT> dataset,
                                               const NodeWorkItem* work_items,
                                               const Split<DataT>* splits,
                                               const WorkloadInfo* workload_info,
                                               const IdxT* partition_row_ids)
{
  const auto workload_info_cta = workload_info[blockIdx.x];
  const auto nid               = workload_info_cta.nodeid;
  const auto work_item         = work_items[nid];
  const auto split             = splits[nid];
  if (SplitNotValid(split, min_impurity_decrease)) { return; }

  const auto range_start = work_item.instances.begin;
  const auto range_len   = work_item.instances.count;
  const auto range_pos   = std::size_t(workload_info_cta.offset_blockid) * blockDim.x + threadIdx.x;
  if (range_pos < range_len) {
    const auto idx       = range_start + range_pos;
    dataset.row_ids[idx] = partition_row_ids[idx];
  }
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launchNodeSplitKernel(const DataT min_impurity_decrease,
                           const Dataset<DataT, LabelT, IdxT>& dataset,
                           const NodeWorkItem* work_items,
                           Split<DataT>* splits,
                           const WorkloadInfo* workload_info,
                           size_t n_blocks_dimx,
                           IdxT* partition_row_ids,
                           cudaStream_t builder_stream)
{
  if (n_blocks_dimx == 0) return;

  using CountT = typename Split<DataT>::CountT;
  nodeSplitLocalCountKernel<DataT, LabelT, IdxT, TPB><<<n_blocks_dimx, TPB, 0, builder_stream>>>(
    min_impurity_decrease, dataset, work_items, splits, workload_info);

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
    if (SplitNotValid(split, min_impurity_decrease)) {
      return NodeSplitPartitionState<CountT>{CountT{0}, false, false};
    }

    const auto range_pos = std::size_t(workload_info_cta.offset_blockid) * TPB + slot % TPB;
    if (range_pos >= work_item.instances.count) {
      return NodeSplitPartitionState<CountT>{CountT{0}, false, false};
    }

    const auto row       = dataset.row_ids[work_item.instances.begin + range_pos];
    const auto col_idx   = std::size_t(split.colid) * dataset.M + row;
    const auto goes_left = dataset.data[col_idx] <= split.quesval;
    return NodeSplitPartitionState<CountT>{goes_left ? CountT{1} : CountT{0}, true, goes_left};
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
                                thrust::equal_to<int>{},
                                NodeSplitPartitionScanOp<CountT>{});

  // The original row_ids buffer remains the source during the scan, so copy back after it finishes.
  nodeSplitCopyBackKernel<DataT, LabelT, IdxT, TPB><<<n_blocks_dimx, TPB, 0, builder_stream>>>(
    min_impurity_decrease, dataset, work_items, splits, workload_info, partition_row_ids);
}

template <typename DatasetT, typename NodeT, typename ObjectiveT>
static __global__ void leafKernel(ObjectiveT objective,
                                  DatasetT dataset,
                                  const NodeT* tree,
                                  const InstanceRange* instance_ranges,
                                  typename ObjectiveT::BinT* leaf_histograms)
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
    auto label = dataset.labels[dataset.row_ids[i]];
    BinT::IncrementHistogram(histogram, 1, 0, label);
  }
  __syncthreads();
  if (tid == 0) {
    auto leaf_histogram = leaf_histograms + dataset.num_outputs * node_id;
    for (int i = 0; i < dataset.num_outputs; ++i) {
      leaf_histogram[i] = histogram[i];
    }
  }
}

template <typename NodeT, typename ObjectiveT, typename DataT>
static __global__ void finalizeLeafKernel(ObjectiveT objective,
                                          const NodeT* tree,
                                          const typename ObjectiveT::BinT* leaf_histograms,
                                          DataT* leaves,
                                          int num_outputs)
{
  auto node_id = blockIdx.x;
  auto leaf    = leaves + num_outputs * node_id;
  auto& node   = tree[node_id];
  if (!node.IsLeaf()) {
    for (int i = 0; i < num_outputs; ++i) {
      leaf[i] = DataT(0);
    }
    return;
  }
  auto leaf_histogram = leaf_histograms + num_outputs * node_id;
  ObjectiveT::SetLeafVector(leaf_histogram, num_outputs, leaf);
}

template <typename NodeT, typename ObjectiveT, typename DataT>
void launchFinalizeLeafKernel(ObjectiveT objective,
                              const NodeT* tree,
                              const typename ObjectiveT::BinT* leaf_histograms,
                              DataT* leaves,
                              int batch_size,
                              int num_outputs,
                              cudaStream_t builder_stream)
{
  finalizeLeafKernel<<<batch_size, 1, 0, builder_stream>>>(
    objective, tree, leaf_histograms, leaves, num_outputs);
}

template <typename DatasetT, typename NodeT, typename ObjectiveT>
void launchLeafHistogramKernel(ObjectiveT objective,
                               DatasetT& dataset,
                               const NodeT* tree,
                               const InstanceRange* instance_ranges,
                               typename ObjectiveT::BinT* leaf_histograms,
                               int batch_size,
                               size_t smem_size,
                               cudaStream_t builder_stream)
{
  int num_blocks = batch_size;
  leafKernel<<<num_blocks, TPB_DEFAULT, smem_size, builder_stream>>>(
    objective, dataset, tree, instance_ranges, leaf_histograms);
}

/**
 * @brief For every threadblock, converts the smem pdf-histogram to
 *        cdf-histogram inplace using inclusive block-sum-scan and returns
 *        the total_sum
 * @return The total sum aggregated over the sumscan,
 *         as well as the modified cdf-histogram pointer
 */
template <typename BinT, typename IdxT, int TPB>
DI BinT pdf_to_cdf(BinT* shared_histogram, IdxT n_bins)
{
  // Blockscan instance preparation
  typedef cub::BlockScan<BinT, TPB> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // variable to accumulate aggregate of sumscans of previous iterations
  BinT total_aggregate = BinT();

  for (IdxT tix = threadIdx.x; tix < raft::ceildiv(n_bins, TPB) * TPB; tix += blockDim.x) {
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

template <typename BinT>
DI BinCountT bin_count(BinT const& bin)
{
  return bin.Count();
}

template <typename DataT, typename LabelT, typename IdxT, int TPB, typename BinT>
static __global__ void computeSplitHistogramKernel(BinT* histograms,
                                                   IdxT max_n_bins,
                                                   const Dataset<DataT, LabelT, IdxT> dataset,
                                                   const Quantiles<DataT, IdxT> quantiles,
                                                   const NodeWorkItem* work_items,
                                                   IdxT colStart,
                                                   const IdxT* column_samples,
                                                   const WorkloadInfo* workload_info)
{
  extern __shared__ char smem[];

  auto workload_info_cta = workload_info[blockIdx.x];
  IdxT nid               = workload_info_cta.nodeid;
  const auto work_item   = work_items[nid];
  auto range_start       = work_item.instances.begin;
  auto range_len         = work_item.instances.count;
  IdxT offset_blockid    = workload_info_cta.offset_blockid;
  IdxT num_blocks        = workload_info_cta.num_blocks;

  IdxT col;
  if (dataset.n_sampled_cols == dataset.N) {
    col = colStart + blockIdx.y;
  } else {
    IdxT colIndex = colStart + blockIdx.y;
    col           = column_samples[nid * dataset.n_sampled_cols + colIndex];
  }

  int n_bins                = quantiles.n_bins_array[col];
  auto shared_histogram_len = n_bins * dataset.num_outputs;
  auto* shared_histogram    = alignPointer<BinT>(smem);
  auto* shared_quantiles    = alignPointer<DataT>(shared_histogram + shared_histogram_len);
  IdxT stride               = blockDim.x * num_blocks;
  IdxT tid                  = threadIdx.x + offset_blockid * blockDim.x;
  auto histograms_offset    = ((nid * gridDim.y) + blockIdx.y) * max_n_bins * dataset.num_outputs;

  for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x) {
    shared_histogram[i] = BinT();
  }
  for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x) {
    shared_quantiles[b] = quantiles.quantiles_array[max_n_bins * col + b];
  }
  __syncthreads();

  std::size_t col_offset = std::size_t(col) * dataset.M;
  for (auto i = range_start + tid; i < range_start + range_len; i += stride) {
    auto row   = dataset.row_ids[i];
    auto data  = dataset.data[row + col_offset];
    auto label = dataset.labels[row];
    IdxT start = lower_bound(shared_quantiles, n_bins, data);
    BinT::IncrementHistogram(shared_histogram, n_bins, start, label);
  }
  __syncthreads();

  for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x) {
    BinT::AtomicAdd(histograms + histograms_offset + i, shared_histogram[i]);
  }
}

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename ObjectiveT,
          typename BinT>
static __global__ void evaluateSplitKernel(BinT* histograms,
                                           IdxT max_n_bins,
                                           const Dataset<DataT, LabelT, IdxT> dataset,
                                           const Quantiles<DataT, IdxT> quantiles,
                                           IdxT colStart,
                                           const IdxT* column_samples,
                                           int* mutex,
                                           volatile Split<DataT>* splits,
                                           ObjectiveT objective)
{
  extern __shared__ char smem[];

  IdxT nid = blockIdx.x;
  IdxT col;
  if (dataset.n_sampled_cols == dataset.N) {
    col = colStart + blockIdx.y;
  } else {
    IdxT colIndex = colStart + blockIdx.y;
    col           = column_samples[nid * dataset.n_sampled_cols + colIndex];
  }

  int n_bins                = quantiles.n_bins_array[col];
  auto shared_histogram_len = n_bins * objective.NumClasses();
  auto* shared_histogram    = alignPointer<BinT>(smem);
  auto* shared_quantiles    = alignPointer<DataT>(shared_histogram + shared_histogram_len);
  auto histograms_offset = ((nid * gridDim.y) + blockIdx.y) * max_n_bins * objective.NumClasses();

  for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x) {
    shared_histogram[i] = histograms[histograms_offset + i];
  }
  for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x) {
    shared_quantiles[b] = quantiles.quantiles_array[max_n_bins * col + b];
  }
  __syncthreads();

  typename ObjectiveT::CountT split_len = 0;
  for (IdxT c = 0; c < objective.NumClasses(); ++c) {
    auto total_sum = pdf_to_cdf<BinT, IdxT, TPB>(shared_histogram + n_bins * c, n_bins);
    split_len += bin_count(total_sum);
  }
  __syncthreads();

  Split<DataT> sp =
    objective.Gain(shared_histogram, shared_quantiles, static_cast<int>(col), split_len, n_bins);
  __syncthreads();
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

template <typename DataT, typename LabelT, typename IdxT, int TPB, typename BinT>
void launchComputeSplitHistogramKernel(BinT* histograms,
                                       IdxT max_n_bins,
                                       const Dataset<DataT, LabelT, IdxT>& dataset,
                                       const Quantiles<DataT, IdxT>& quantiles,
                                       const NodeWorkItem* work_items,
                                       IdxT colStart,
                                       const IdxT* column_samples,
                                       const WorkloadInfo* workload_info,
                                       dim3 grid,
                                       size_t smem_size,
                                       cudaStream_t builder_stream)
{
  computeSplitHistogramKernel<DataT, LabelT, IdxT, TPB_DEFAULT>
    <<<grid, TPB_DEFAULT, smem_size, builder_stream>>>(histograms,
                                                       max_n_bins,
                                                       dataset,
                                                       quantiles,
                                                       work_items,
                                                       colStart,
                                                       column_samples,
                                                       workload_info);
}

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename ObjectiveT,
          typename BinT>
void launchEvaluateSplitKernel(BinT* histograms,
                               IdxT max_n_bins,
                               const Dataset<DataT, LabelT, IdxT>& dataset,
                               const Quantiles<DataT, IdxT>& quantiles,
                               IdxT colStart,
                               const IdxT* column_samples,
                               int* mutex,
                               volatile Split<DataT>* splits,
                               ObjectiveT& objective,
                               dim3 grid,
                               size_t smem_size,
                               cudaStream_t builder_stream)
{
  evaluateSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>
    <<<grid, TPB_DEFAULT, smem_size, builder_stream>>>(histograms,
                                                       max_n_bins,
                                                       dataset,
                                                       quantiles,
                                                       colStart,
                                                       column_samples,
                                                       mutex,
                                                       splits,
                                                       objective);
}

template void launchNodeSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT>(
  const _DataT min_impurity_decrease,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const NodeWorkItem* work_items,
  Split<_DataT>* splits,
  const WorkloadInfo* workload_info,
  size_t n_blocks_dimx,
  _IdxT* partition_row_ids,
  cudaStream_t builder_stream);

template void launchLeafHistogramKernel<_DatasetT, _NodeT, _ObjectiveT>(
  _ObjectiveT objective,
  _DatasetT& dataset,
  const _NodeT* tree,
  const InstanceRange* instance_ranges,
  typename _ObjectiveT::BinT* leaf_histograms,
  int batch_size,
  size_t smem_size,
  cudaStream_t builder_stream);

template void launchFinalizeLeafKernel<_NodeT, _ObjectiveT, _DataT>(
  _ObjectiveT objective,
  const _NodeT* tree,
  const typename _ObjectiveT::BinT* leaf_histograms,
  _DataT* leaves,
  int batch_size,
  int num_outputs,
  cudaStream_t builder_stream);

template void launchComputeSplitHistogramKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT, _BinT>(
  _BinT* histograms,
  _IdxT max_n_bins,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const Quantiles<_DataT, _IdxT>& quantiles,
  const NodeWorkItem* work_items,
  _IdxT colStart,
  const _IdxT* column_samples,
  const WorkloadInfo* workload_info,
  dim3 grid,
  size_t smem_size,
  cudaStream_t builder_stream);

template void launchEvaluateSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT, _ObjectiveT, _BinT>(
  _BinT* histograms,
  _IdxT max_n_bins,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const Quantiles<_DataT, _IdxT>& quantiles,
  _IdxT colStart,
  const _IdxT* column_samples,
  int* mutex,
  volatile Split<_DataT>* splits,
  _ObjectiveT& objective,
  dim3 grid,
  size_t smem_size,
  cudaStream_t builder_stream);
}  // namespace DT
}  // namespace ML
