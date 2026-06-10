/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "builder_kernels.cuh"

#include <common/grid_sync.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>

#include <algorithm>
#include <cstdio>

namespace ML {
namespace DT {

static constexpr int TPB_DEFAULT = 128;

/**
 * @brief Partition the samples to left/right nodes based on the best split
 * @return the position of the left child node in the nodes list. However, this
 *         value is valid only for threadIdx.x == 0.
 * @note this should be called by only one block from all participating blocks
 *       'smem' should be at least of size `sizeof(IdxT) * TPB * 2`
 */
template <typename DataT, typename LabelT, typename IdxT, int TPB>
DI void partitionSamples(const Dataset<DataT, LabelT, IdxT>& dataset,
                         const Split<DataT>& split,
                         const NodeWorkItem& work_item,
                         char* smem)
{
  typedef cub::BlockScan<int, TPB> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp1, temp2;
  volatile auto* row_ids = reinterpret_cast<volatile IdxT*>(dataset.row_ids);
  // for compaction
  size_t smemSize  = sizeof(IdxT) * TPB;
  auto* lcomp      = reinterpret_cast<IdxT*>(smem);
  auto* rcomp      = reinterpret_cast<IdxT*>(smem + smemSize);
  auto range_start = work_item.instances.begin;
  auto range_len   = work_item.instances.count;
  auto* col        = dataset.data + split.colid * std::size_t(dataset.M);
  std::size_t loffset = range_start;
  std::size_t part    = loffset + std::size_t(split.local_nLeft);
  std::size_t roffset = part;
  auto end  = range_start + range_len;
  int lflag = 0, rflag = 0, llen = 0, rlen = 0, minlen = 0;
  auto tid = threadIdx.x;
  while (loffset < part && roffset < end) {
    // find the samples in the left that belong to right and vice-versa
    auto loff = loffset + tid, roff = roffset + tid;
    if (llen == minlen) lflag = loff < part ? col[row_ids[loff]] > split.quesval : 0;
    if (rlen == minlen) rflag = roff < end ? col[row_ids[roff]] <= split.quesval : 0;
    // scan to compute the locations for each 'misfit' in the two partitions
    int lidx, ridx;
    BlockScanT(temp1).ExclusiveSum(lflag, lidx, llen);
    BlockScanT(temp2).ExclusiveSum(rflag, ridx, rlen);
    __syncthreads();
    minlen = llen < rlen ? llen : rlen;
    // compaction to figure out the right locations to swap
    if (lflag) lcomp[lidx] = loff;
    if (rflag) rcomp[ridx] = roff;
    __syncthreads();
    // reset the appropriate flags for the longer of the two
    if (lidx < minlen) lflag = 0;
    if (ridx < minlen) rflag = 0;
    if (llen == minlen) loffset += TPB;
    if (rlen == minlen) roffset += TPB;
    // swap the 'misfit's
    if (tid < minlen) {
      auto a              = row_ids[lcomp[tid]];
      auto b              = row_ids[rcomp[tid]];
      row_ids[lcomp[tid]] = b;
      row_ids[rcomp[tid]] = a;
    }
  }
}
template <typename DataT, typename LabelT, typename IdxT, int TPB>
static __global__ void nodeSplitKernel(const DataT min_impurity_decrease,
                                       const Dataset<DataT, LabelT, IdxT> dataset,
                                       const NodeWorkItem* work_items,
                                       Split<DataT>* splits)
{
  extern __shared__ char smem[];
  const auto work_item = work_items[blockIdx.x];
  auto split           = splits[blockIdx.x];
  if (split.best_metric_val <= min_impurity_decrease) { return; }

  using CountT     = typename Split<DataT>::CountT;
  auto* left_count = reinterpret_cast<CountT*>(smem);
  if (threadIdx.x == 0) { *left_count = CountT{0}; }
  __syncthreads();

  auto* col = dataset.data + split.colid * std::size_t(dataset.M);
  for (auto i = work_item.instances.begin + threadIdx.x;
       i < work_item.instances.begin + work_item.instances.count;
       i += blockDim.x) {
    auto row = dataset.row_ids[i];
    if (col[row] <= split.quesval) { atomicAdd(left_count, CountT{1}); }
  }
  __syncthreads();

  split.local_nLeft = *left_count;
  if (threadIdx.x == 0) { splits[blockIdx.x].local_nLeft = split.local_nLeft; }
  auto* partition_smem = alignPointer<char>(left_count + 1);
  partitionSamples<DataT, LabelT, IdxT, TPB>(dataset, split, work_item, partition_smem);
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launchNodeSplitKernel(const DataT min_impurity_decrease,
                           const Dataset<DataT, LabelT, IdxT>& dataset,
                           const NodeWorkItem* work_items,
                           const size_t work_items_size,
                           Split<DataT>* splits,
                           cudaStream_t builder_stream)
{
  using CountT             = typename Split<DataT>::CountT;
  auto constexpr smem_size = sizeof(CountT) + 2 * sizeof(IdxT) * TPB + sizeof(IdxT);
  nodeSplitKernel<DataT, LabelT, IdxT, TPB>
    <<<work_items_size, TPB, smem_size, builder_stream>>>(min_impurity_decrease,
                                                          dataset,
                                                          work_items,
                                                          splits);
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

DI unsigned long long int bin_count(CountBin const& bin) { return bin.x; }

DI unsigned long long int bin_count(AggregateBin const& bin) { return bin.count; }

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename BinT>
static __global__ void computeSplitHistogramKernel(BinT* histograms,
                                                   IdxT max_n_bins,
                                                   const Dataset<DataT, LabelT, IdxT> dataset,
                                                   const Quantiles<DataT, IdxT> quantiles,
                                                   const NodeWorkItem* work_items,
                                                   IdxT colStart,
                                                   const IdxT* colids,
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
    col           = colids[nid * dataset.n_sampled_cols + colIndex];
  }

  int n_bins                = quantiles.n_bins_array[col];
  auto shared_histogram_len = n_bins * dataset.num_outputs;
  auto* shared_histogram    = alignPointer<BinT>(smem);
  auto* shared_quantiles    = alignPointer<DataT>(shared_histogram + shared_histogram_len);
  IdxT stride               = blockDim.x * num_blocks;
  IdxT tid                  = threadIdx.x + offset_blockid * blockDim.x;
  auto histograms_offset = ((nid * gridDim.y) + blockIdx.y) * max_n_bins * dataset.num_outputs;

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
                                           const IdxT* colids,
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
    col           = colids[nid * dataset.n_sampled_cols + colIndex];
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

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename BinT>
void launchComputeSplitHistogramKernel(BinT* histograms,
                                       IdxT max_n_bins,
                                       const Dataset<DataT, LabelT, IdxT>& dataset,
                                       const Quantiles<DataT, IdxT>& quantiles,
                                       const NodeWorkItem* work_items,
                                       IdxT colStart,
                                       const IdxT* colids,
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
                                                       colids,
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
                               const IdxT* colids,
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
                                                       colids,
                                                       mutex,
                                                       splits,
                                                       objective);
}

template void launchNodeSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT>(
  const _DataT min_impurity_decrease,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const NodeWorkItem* work_items,
  const size_t work_items_size,
  Split<_DataT>* splits,
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

template void
launchComputeSplitHistogramKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT, _BinT>(
  _BinT* histograms,
  _IdxT max_n_bins,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const Quantiles<_DataT, _IdxT>& quantiles,
  const NodeWorkItem* work_items,
  _IdxT colStart,
  const _IdxT* colids,
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
  const _IdxT* colids,
  int* mutex,
  volatile Split<_DataT>* splits,
  _ObjectiveT& objective,
  dim3 grid,
  size_t smem_size,
  cudaStream_t builder_stream);
}  // namespace DT
}  // namespace ML
