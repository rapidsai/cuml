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
  auto tid                     = threadIdx.x;
  constexpr bool kIsClassifier = std::is_same<BinT, CountBin>::value;
  constexpr bool kIsRegressor  = std::is_same<BinT, AggregateBin>::value;
  static_assert(kIsClassifier || kIsRegressor, "unknown BinT in leafKernel");
  // Regressor leaf mean is sum(label*weight) / sum(weight); accumulate the
  // weighted denominator once in shared memory and hand it to SetLeafVector.
  // Classifier ignores the value.
  __shared__ double shared_weight_at_leaf;
  if constexpr (kIsRegressor) {
    if (tid == 0) shared_weight_at_leaf = 0.0;
  }
  for (int i = tid; i < dataset.num_outputs; i += blockDim.x) {
    histogram[i] = BinT();
  }
  __syncthreads();
  bool has_weight = (dataset.sample_weight != nullptr);
  for (auto i = range.begin + tid; i < range.begin + range.count; i += blockDim.x) {
    auto row      = dataset.row_ids[i];
    auto label    = dataset.labels[row];
    double weight = has_weight ? static_cast<double>(dataset.sample_weight[row]) : 1.0;
    BinT::IncrementHistogram(histogram, 1, 0, label, weight);
    if constexpr (kIsRegressor) { atomicAdd(&shared_weight_at_leaf, weight); }
  }
  __syncthreads();
  if (tid == 0) {
    double weighted_total = 0.0;
    if constexpr (kIsRegressor) { weighted_total = shared_weight_at_leaf; }
    ObjectiveT::SetLeafVector(
      histogram, dataset.num_outputs, leaves + dataset.num_outputs * node_id, weighted_total);
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
                                          int* unweighted_histograms,
                                          double* weighted_count_histograms,
                                          IdxT max_n_bins,
                                          IdxT min_samples_split,
                                          IdxT max_leaves,
                                          const Dataset<DataT, LabelT, IdxT> dataset,
                                          const Quantiles<DataT, IdxT> quantiles,
                                          const NodeWorkItem* work_items,
                                          IdxT colStart,
                                          const IdxT* colids,
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
  IdxT col;
  if (dataset.n_sampled_cols == dataset.N) {
    col = colStart + blockIdx.y;
  } else {
    IdxT colIndex = colStart + blockIdx.y;
    col           = colids[nid * dataset.n_sampled_cols + colIndex];
  }

  // getting the n_bins for that feature
  int n_bins = quantiles.n_bins_array[col];

  constexpr bool kIsClassifier = std::is_same<BinT, CountBin>::value;
  constexpr bool kIsRegressor  = std::is_same<BinT, AggregateBin>::value;
  static_assert(kIsClassifier || kIsRegressor, "unknown BinT in computeSplitKernel");

  auto end                      = range_start + range_len;
  auto shared_histogram_len     = n_bins * objective.NumClasses();
  auto* shared_histogram        = alignPointer<BinT>(smem);
  int* shared_unweighted        = nullptr;
  double* shared_weighted_count = nullptr;
  DataT* shared_quantiles;
  int* shared_done;
  if constexpr (kIsClassifier) {
    shared_unweighted = alignPointer<int>(shared_histogram + shared_histogram_len);
    shared_quantiles  = alignPointer<DataT>(shared_unweighted + n_bins);
  } else if constexpr (kIsRegressor) {
    shared_weighted_count = alignPointer<double>(shared_histogram + shared_histogram_len);
    shared_quantiles      = alignPointer<DataT>(shared_weighted_count + n_bins);
  } else {
    shared_quantiles = alignPointer<DataT>(shared_histogram + shared_histogram_len);
  }
  shared_done = alignPointer<int>(shared_quantiles + n_bins);
  IdxT stride = blockDim.x * num_blocks;
  IdxT tid    = threadIdx.x + offset_blockid * blockDim.x;

  // populating shared memory with initial values
  for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x)
    shared_histogram[i] = BinT();
  for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x) {
    if constexpr (kIsClassifier) shared_unweighted[b] = 0;
    if constexpr (kIsRegressor) shared_weighted_count[b] = 0.0;
    shared_quantiles[b] = quantiles.quantiles_array[max_n_bins * col + b];
  }

  // synchronizing above changes across block
  __syncthreads();

  bool has_weight = (dataset.sample_weight != nullptr);
  // Must be 64 bit - can easily grow larger than a 32 bit int
  std::size_t col_offset = std::size_t(col) * dataset.M;
  for (auto i = range_start + tid; i < end; i += stride) {
    // each thread works over a data point and strides to the next
    auto row      = dataset.row_ids[i];
    auto data     = dataset.data[row + col_offset];
    auto label    = dataset.labels[row];
    double weight = has_weight ? static_cast<double>(dataset.sample_weight[row]) : 1.0;

    // `start` is lowest index such that data <= shared_quantiles[start]
    IdxT start = lower_bound(shared_quantiles, n_bins, data);
    BinT::IncrementHistogram(shared_histogram, n_bins, start, label, weight);
    if constexpr (kIsClassifier) atomicAdd(&shared_unweighted[start], 1);
    if constexpr (kIsRegressor) atomicAdd(&shared_weighted_count[start], weight);
  }

  // synchronizing above changes across block
  __syncthreads();
  if (num_blocks > 1) {
    // update the corresponding global location
    auto histograms_offset =
      ((large_nid * gridDim.y) + blockIdx.y) * max_n_bins * objective.NumClasses();
    auto companion_offset = ((large_nid * gridDim.y) + blockIdx.y) * max_n_bins;
    for (IdxT i = threadIdx.x; i < shared_histogram_len; i += blockDim.x) {
      BinT::AtomicAdd(histograms + histograms_offset + i, shared_histogram[i]);
    }
    if constexpr (kIsClassifier) {
      for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x) {
        atomicAdd(&unweighted_histograms[companion_offset + b], shared_unweighted[b]);
      }
    } else if constexpr (kIsRegressor) {
      for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x) {
        atomicAdd(&weighted_count_histograms[companion_offset + b], shared_weighted_count[b]);
      }
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
    if constexpr (kIsClassifier) {
      for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x)
        shared_unweighted[b] = unweighted_histograms[companion_offset + b];
    } else if constexpr (kIsRegressor) {
      for (IdxT b = threadIdx.x; b < n_bins; b += blockDim.x)
        shared_weighted_count[b] = weighted_count_histograms[companion_offset + b];
    }

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
  if constexpr (kIsClassifier) { pdf_to_cdf<int, IdxT, TPB>(shared_unweighted, n_bins); }
  if constexpr (kIsRegressor) { pdf_to_cdf<double, IdxT, TPB>(shared_weighted_count, n_bins); }

  __syncthreads();

  // calculate the best candidate bins (one for each thread in the block) in current feature and
  // corresponding information gain for splitting
  Split<DataT, IdxT> sp;
  if constexpr (kIsClassifier) {
    sp =
      objective.Gain(shared_histogram, shared_quantiles, col, range_len, n_bins, shared_unweighted);
  } else {
    sp = objective.Gain(
      shared_histogram, shared_quantiles, col, range_len, n_bins, shared_weighted_count);
  }

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
                              int* unweighted_histograms,
                              double* weighted_count_histograms,
                              IdxT max_n_bins,
                              IdxT min_samples_split,
                              IdxT max_leaves,
                              const Dataset<DataT, LabelT, IdxT>& dataset,
                              const Quantiles<DataT, IdxT>& quantiles,
                              const NodeWorkItem* work_items,
                              IdxT colStart,
                              const IdxT* colids,
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
                                                       unweighted_histograms,
                                                       weighted_count_histograms,
                                                       max_n_bins,
                                                       min_samples_split,
                                                       max_leaves,
                                                       dataset,
                                                       quantiles,
                                                       work_items,
                                                       colStart,
                                                       colids,
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
  int* unweighted_histograms,
  double* weighted_count_histograms,
  _IdxT n_bins,
  _IdxT min_samples_split,
  _IdxT max_leaves,
  const Dataset<_DataT, _LabelT, _IdxT>& dataset,
  const Quantiles<_DataT, _IdxT>& quantiles,
  const NodeWorkItem* work_items,
  _IdxT colStart,
  const _IdxT* colids,
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
