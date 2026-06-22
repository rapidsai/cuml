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
                         const Split<DataT, IdxT>& split,
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
  auto loffset = range_start, part = loffset + split.nLeft, roffset = part;
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
static __global__ void nodeSplitKernel(const IdxT min_samples_leaf,
                                       const IdxT min_samples_split,
                                       const IdxT max_leaves,
                                       const DataT min_impurity_decrease,
                                       const Dataset<DataT, LabelT, IdxT> dataset,
                                       const NodeWorkItem* work_items,
                                       const Split<DataT, IdxT>* splits)
{
  extern __shared__ char smem[];
  const auto work_item = work_items[blockIdx.x];
  const auto split     = splits[blockIdx.x];
  if (SplitNotValid(
        split, min_impurity_decrease, min_samples_leaf, IdxT(work_item.instances.count))) {
    return;
  }
  partitionSamples<DataT, LabelT, IdxT, TPB>(dataset, split, work_item, (char*)smem);
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launchNodeSplitKernel(const IdxT min_samples_leaf,
                           const IdxT min_samples_split,
                           const IdxT max_leaves,
                           const DataT min_impurity_decrease,
                           const Dataset<DataT, LabelT, IdxT>& dataset,
                           const NodeWorkItem* work_items,
                           const size_t work_items_size,
                           const Split<DataT, IdxT>* splits,
                           cudaStream_t builder_stream)
{
  auto constexpr smem_size = 2 * sizeof(IdxT) * TPB;
  nodeSplitKernel<DataT, LabelT, IdxT, TPB>
    <<<work_items_size, TPB, smem_size, builder_stream>>>(min_samples_leaf,
                                                          min_samples_split,
                                                          max_leaves,
                                                          min_impurity_decrease,
                                                          dataset,
                                                          work_items,
                                                          splits);
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
static __global__ void computeSplitBoundaryKernel(const IdxT min_samples_leaf,
                                                  const DataT min_impurity_decrease,
                                                  const Dataset<DataT, LabelT, IdxT> dataset,
                                                  const NodeWorkItem* work_items,
                                                  const Split<DataT, IdxT>* splits,
                                                  DataT* split_left_max,
                                                  DataT* split_right_min)
{
  using BlockReduce = cub::BlockReduce<DataT, TPB>;
  __shared__ typename BlockReduce::TempStorage left_storage;
  __shared__ typename BlockReduce::TempStorage right_storage;

  const auto work_item = work_items[blockIdx.x];
  const auto split     = splits[blockIdx.x];
  const DataT lowest   = -std::numeric_limits<DataT>::max();
  const DataT highest  = std::numeric_limits<DataT>::max();

  DataT thread_left_max  = lowest;
  DataT thread_right_min = highest;

  if (!SplitNotValid(
        split, min_impurity_decrease, min_samples_leaf, IdxT(work_item.instances.count))) {
    auto* col = dataset.data + split.colid * std::size_t(dataset.M);
    auto begin = work_item.instances.begin;
    auto end   = begin + work_item.instances.count;
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
      auto data = col[dataset.row_ids[i]];
      if (data <= split.quesval) {
        thread_left_max = max(thread_left_max, data);
      } else {
        thread_right_min = min(thread_right_min, data);
      }
    }
  }

  auto max_op = [] __device__(DataT a, DataT b) { return max(a, b); };
  auto min_op = [] __device__(DataT a, DataT b) { return min(a, b); };
  DataT left_max = BlockReduce(left_storage).Reduce(thread_left_max, max_op);
  __syncthreads();
  DataT right_min = BlockReduce(right_storage).Reduce(thread_right_min, min_op);

  if (threadIdx.x == 0) {
    split_left_max[blockIdx.x]  = left_max;
    split_right_min[blockIdx.x] = right_min;
  }
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launchComputeSplitBoundaryKernel(const IdxT min_samples_leaf,
                                      const DataT min_impurity_decrease,
                                      const Dataset<DataT, LabelT, IdxT>& dataset,
                                      const NodeWorkItem* work_items,
                                      const size_t work_items_size,
                                      const Split<DataT, IdxT>* splits,
                                      DataT* split_left_max,
                                      DataT* split_right_min,
                                      cudaStream_t builder_stream)
{
  computeSplitBoundaryKernel<DataT, LabelT, IdxT, TPB>
    <<<work_items_size, TPB, 0, builder_stream>>>(min_samples_leaf,
                                                  min_impurity_decrease,
                                                  dataset,
                                                  work_items,
                                                  splits,
                                                  split_left_max,
                                                  split_right_min);
}

template <typename DataT, typename IdxT, int TPB>
static __global__ void applySplitThresholdRefinementKernel(const size_t work_items_size,
                                                           Split<DataT, IdxT>* splits,
                                                           const DataT* split_left_max,
                                                           const DataT* split_right_min,
                                                           const Quantiles<DataT, IdxT> quantiles,
                                                           IdxT max_n_bins)
{
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < work_items_size;
       i += blockDim.x * gridDim.x) {
    auto left_max       = split_left_max[i];
    auto right_min      = split_right_min[i];
    const DataT lowest  = -std::numeric_limits<DataT>::max();
    const DataT highest = std::numeric_limits<DataT>::max();
    if (splits[i].colid != IdxT(-1) && left_max != lowest && right_min != highest &&
        left_max < right_min) {
      DataT threshold = left_max + (right_min - left_max) / DataT(2.0);
      auto colid      = splits[i].colid;
      auto n_bins     = quantiles.n_bins_array[colid];
      auto col_quantiles =
        quantiles.quantiles_array + std::size_t(colid) * std::size_t(max_n_bins);
      if (n_bins > 1) {
        auto left_bin  = lower_bound(col_quantiles, n_bins, left_max);
        auto right_bin = lower_bound(col_quantiles, n_bins, right_min);
        if (right_bin > left_bin + 1) {
          auto middle_empty_bin = left_bin + (right_bin - left_bin) / 2;
          threshold             = col_quantiles[middle_empty_bin];
        }
      }
      splits[i].pred_quesval = threshold;
    } else {
      splits[i].pred_quesval = splits[i].quesval;
    }
  }
}

template <typename DataT, typename IdxT, int TPB>
void launchApplySplitThresholdRefinementKernel(const size_t work_items_size,
                                               Split<DataT, IdxT>* splits,
                                               const DataT* split_left_max,
                                               const DataT* split_right_min,
                                               const Quantiles<DataT, IdxT>& quantiles,
                                               IdxT max_n_bins,
                                               cudaStream_t builder_stream)
{
  int num_blocks = raft::ceildiv<int>(work_items_size, TPB);
  applySplitThresholdRefinementKernel<DataT, IdxT, TPB><<<num_blocks, TPB, 0, builder_stream>>>(
    work_items_size, splits, split_left_max, split_right_min, quantiles, max_n_bins);
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
    auto label = dataset.labels[dataset.row_ids[i]];
    BinT::IncrementHistogram(histogram, 1, 0, label);
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

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename ObjectiveT,
          typename BinT>
static __global__ void computeSplitKernel(BinT* histograms,
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
                                          uint64_t seed)
{
  // dynamic shared memory
  extern __shared__ char smem[];

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

  auto end                  = range_start + range_len;
  auto shared_histogram_len = n_bins * objective.NumClasses();
  auto* shared_histogram    = alignPointer<BinT>(smem);
  auto* shared_quantiles    = alignPointer<DataT>(shared_histogram + shared_histogram_len);
  auto* shared_done         = alignPointer<int>(shared_quantiles + n_bins);
  IdxT stride               = blockDim.x * num_blocks;
  IdxT tid                  = threadIdx.x + offset_blockid * blockDim.x;

  // populating shared memory with initial values
  for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x)
    shared_histogram[i] = BinT();
  for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x)
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

    // `start` is lowest index such that data <= shared_quantiles[start]
    IdxT start = lower_bound(shared_quantiles, n_bins, data);
    // ++shared_histogram[start]
    BinT::IncrementHistogram(shared_histogram, n_bins, start, label);
  }

  // synchronizing above changes across block
  __syncthreads();
  if (num_blocks > 1) {
    // update the corresponding global location
    auto histograms_offset =
      ((large_nid * gridDim.y) + blockIdx.y) * max_n_bins * objective.NumClasses();
    for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x) {
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
    for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x)
      shared_histogram[i] = histograms[histograms_offset + i];

    __syncthreads();
  }

  // PDF to CDF inplace in `shared_histogram`
  for (IdxT c = 0; c < objective.NumClasses(); ++c) {
    // left to right scan operation for scanning
    // "lesser-than-or-equal" counts
    BinT total_sum = pdf_to_cdf<BinT, IdxT, TPB>(shared_histogram + n_bins * c, n_bins);
    // now, `shared_histogram[n_bins * c + i]` will have count of datapoints of class `c`
    // that are less than or equal to `shared_quantiles[i]`.
  }

  __syncthreads();

  // calculate the best candidate bins (one for each thread in the block) in current feature and
  // corresponding information gain for splitting
  Split<DataT, IdxT> sp =
    objective.Gain(shared_histogram, shared_quantiles, col, colIndex, range_len, n_bins);

  __syncthreads();

  // calculate best bins among candidate bins per feature using warp reduce
  // then atomically update across features to get best split per node
  // (in split[nid])
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename ObjectiveT,
          typename BinT>
void launchComputeSplitKernel(BinT* histograms,
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
                              dim3 grid,
                              size_t smem_size,
                              cudaStream_t builder_stream)
{
  computeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT>
    <<<grid, TPB_DEFAULT, smem_size, builder_stream>>>(histograms,
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

template void launchNodeSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT>(
  const _IdxT min_samples_leaf,
  const _IdxT min_samples_split,
  const _IdxT max_leaves,
  const _DataT min_impurity_decrease,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const NodeWorkItem* work_items,
  const size_t work_items_size,
  const Split<_DataT, _IdxT>* splits,
  cudaStream_t builder_stream);

template void launchComputeSplitBoundaryKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT>(
  const _IdxT min_samples_leaf,
  const _DataT min_impurity_decrease,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const NodeWorkItem* work_items,
  const size_t work_items_size,
  const Split<_DataT, _IdxT>* splits,
  _DataT* split_left_max,
  _DataT* split_right_min,
  cudaStream_t builder_stream);

template void launchApplySplitThresholdRefinementKernel<_DataT, _IdxT, TPB_DEFAULT>(
  const size_t work_items_size,
  Split<_DataT, _IdxT>* splits,
  const _DataT* split_left_max,
  const _DataT* split_right_min,
  const Quantiles<_DataT, _IdxT>& quantiles,
  _IdxT max_n_bins,
  cudaStream_t builder_stream);

template void launchLeafKernel<_DatasetT, _NodeT, _ObjectiveT, _DataT>(
  _ObjectiveT objective,
  _DatasetT& dataset,
  const _NodeT* tree,
  const InstanceRange* instance_ranges,
  _DataT* leaves,
  int batch_size,
  size_t smem_size,
  cudaStream_t builder_stream);

template void launchComputeSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT, _ObjectiveT, _BinT>(
  _BinT* histograms,
  _IdxT n_bins,
  _IdxT min_samples_split,
  _IdxT max_leaves,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const Quantiles<_DataT, _IdxT>& quantiles,
  const NodeWorkItem* work_items,
  _IdxT colStart,
  const _IdxT* column_samples,
  int* done_count,
  int* mutex,
  volatile Split<_DataT, _IdxT>* splits,
  _ObjectiveT& objective,
  _IdxT treeid,
  const WorkloadInfo<_IdxT>* workload_info,
  uint64_t seed,
  dim3 grid,
  size_t smem_size,
  cudaStream_t builder_stream);
}  // namespace DT
}  // namespace ML
