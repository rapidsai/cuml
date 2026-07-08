/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../bins.cuh"
#include "../objectives.cuh"
#include "../quantiles.h"
#include "../random_utils.cuh"

#include <cuml/common/utils.hpp>

#include <cuda/iterator>
#include <cuda/std/random>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>

namespace ML {
namespace DT {

// The range of instances belonging to a particular node
// This structure refers to a range in the device array dataset.row_ids
struct InstanceRange {
  std::size_t begin;
  std::size_t count;
};

struct NodeWorkItem {
  size_t idx;  // Index of the work item in the tree
  int depth;
  InstanceRange instances;
};

/**
 * This struct has information about workload of a single threadblock of
 * computeSplit kernels of classification and regression
 */
template <typename IdxT>
struct WorkloadInfo {
  IdxT nodeid;        // Node in the batch on which the threadblock needs to work
  IdxT large_nodeid;  // counts only large nodes (nodes that require more than one block along x-dim
                      // for histogram calculation)
  IdxT offset_blockid;  // Offset threadblock id among all the blocks that are
                        // working on this node
  IdxT num_blocks;      // Total number of blocks that are working on the node
};

template <typename SplitT, typename IdxT>
HDI bool SplitPartitionNotValid(const SplitT& split, IdxT min_samples_leaf, std::size_t num_rows)
{
  const auto local_count = static_cast<std::int64_t>(num_rows);
  const auto min_leaf    = static_cast<std::int64_t>(min_samples_leaf);
  return split.colid == IdxT(-1) || split.local_nLeft > local_count ||
         split.local_nLeft < min_leaf || (local_count - split.local_nLeft) < min_leaf;
}

template <typename SplitT, typename DataT, typename IdxT>
HDI bool SplitNotValid(const SplitT& split,
                       DataT min_impurity_decrease,
                       IdxT min_samples_leaf,
                       std::size_t num_rows)
{
  return split.best_metric_val <= min_impurity_decrease ||
         SplitPartitionNotValid(split, min_samples_leaf, num_rows);
}

/* Returns 'dataset' rounded up to a correctly-aligned pointer of type OutT* */
template <typename OutT, typename InT>
DI OutT* alignPointer(InT dataset)
{
  return reinterpret_cast<OutT*>(raft::alignTo(reinterpret_cast<size_t>(dataset), sizeof(OutT)));
}

template <typename IdxT>
void sample_features(IdxT* column_samples,
                     const NodeWorkItem* work_items,
                     size_t work_items_size,
                     IdxT treeid,
                     uint64_t seed,
                     IdxT sample_offset,
                     IdxT n,
                     IdxT k,
                     cudaStream_t stream)
{
  auto n_column_samples = work_items_size * size_t(k);
  auto counting         = thrust::make_counting_iterator<size_t>(0);

  thrust::for_each(thrust::cuda::par.on(stream),
                   counting,
                   counting + n_column_samples,
                   [=] __device__(size_t sample_idx) {
                     auto node_idx     = sample_idx / size_t(k);
                     IdxT column_index = static_cast<IdxT>(sample_idx % size_t(k));

                     const uint32_t nodeid = work_items[node_idx].idx;
                     uint32_t rng_seed     = fnv1a32_hash(seed, treeid, nodeid);

                     cuda::shuffle_iterator<IdxT> shuffled_features(
                       n, cuda::std::minstd_rand(rng_seed), sample_offset);
                     column_samples[sample_idx] = shuffled_features[column_index];
                   });
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
                           cudaStream_t builder_stream);

template <typename DatasetT, typename NodeT, typename ObjectiveT, typename DataT>
void launchLeafKernel(ObjectiveT objective,
                      DatasetT& dataset,
                      const NodeT* tree,
                      const InstanceRange* instance_ranges,
                      DataT* leaves,
                      int batch_size,
                      size_t smem_size,
                      cudaStream_t builder_stream);
// Returns the lowest index in `array` whose value is greater or equal to `element`.
// Values outside the quantile range are clamped to the edge bins: values below the
// first quantile return 0, and values above the last quantile return len - 1.
template <typename DataT, typename IdxT>
HDI IdxT lower_bound(DataT const* array, IdxT len, DataT element)
{
  IdxT start = 0;
  IdxT end   = len - 1;
  IdxT mid;
  while (start < end) {
    mid = (start + end) / 2;
    if (array[mid] < element) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename DataT, typename LabelT, typename IdxT, int TPB, typename ObjectiveT>
void launchComputeSplitKernel(typename ObjectiveT::BinT* histograms,
                              IdxT n_bins,
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
                              cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
