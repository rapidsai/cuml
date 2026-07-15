/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../bins.cuh"
#include "../objectives.cuh"
#include "../quantiles.h"
#include "../random_utils.cuh"

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/utils.hpp>

#include <raft/core/error.hpp>

#include <cuda/iterator>
#include <cuda/std/random>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>
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
struct WorkloadInfo {
  std::int64_t nodeid;          // Node in the batch on which the threadblock needs to work
  std::int64_t offset_blockid;  // Offset threadblock id among all the blocks that are
                                // working on this node
  std::int64_t num_blocks;      // Total number of blocks that are working on the node
};

struct SharedMemoryConfig {
  bool use_global_memory_histogram;
  size_t histogram_dynamic_smem_size;
};

template <typename SplitT>
HDI bool SplitPartitionNotValid(const SplitT& split,
                                std::int64_t min_samples_leaf,
                                std::size_t num_rows)
{
  const auto local_count = static_cast<std::int64_t>(num_rows);
  return split.colid == std::int64_t{-1} || split.local_nLeft > local_count ||
         split.local_nLeft < min_samples_leaf ||
         (local_count - split.local_nLeft) < min_samples_leaf;
}

template <typename SplitT, typename DataT>
HDI bool SplitNotValid(const SplitT& split,
                       DataT min_impurity_decrease,
                       std::int64_t min_samples_leaf,
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

inline void sample_features(std::int64_t* column_samples,
                            const NodeWorkItem* work_items,
                            size_t work_items_size,
                            std::int64_t treeid,
                            uint64_t seed,
                            std::int64_t sample_offset,
                            std::int64_t n,
                            std::int64_t k,
                            cudaStream_t stream)
{
  RAFT_EXPECTS(k >= 0, "k must be non-negative");
  RAFT_EXPECTS(n >= k, "k must not exceed n");

  auto sampled_cols     = ML::narrow_cast<std::size_t>(k);
  auto n_column_samples = ML::checked_mul<std::size_t>(work_items_size, sampled_cols);
  auto counting         = thrust::make_counting_iterator<std::size_t>(0);

  thrust::for_each(thrust::cuda::par.on(stream),
                   counting,
                   counting + n_column_samples,
                   [=] __device__(std::size_t sample_idx) {
                     auto node_idx     = sample_idx / sampled_cols;
                     auto column_index = static_cast<std::int64_t>(sample_idx % sampled_cols);

                     auto nodeid       = work_items[node_idx].idx;
                     uint32_t rng_seed = fnv1a32_hash(seed, treeid, nodeid);

                     cuda::shuffle_iterator<std::int64_t> shuffled_features(
                       n, cuda::std::minstd_rand(rng_seed), sample_offset);
                     column_samples[sample_idx] = shuffled_features[column_index];
                   });
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
template <typename DataT, typename LabelT, int TPB, typename ObjectiveT>
void launchComputeSplitKernels(typename ObjectiveT::BinT* histograms,
                               std::int64_t n_bins,
                               const Dataset<DataT, LabelT>& dataset,
                               const Quantiles<DataT>& quantiles,
                               const NodeWorkItem* work_items,
                               std::int64_t colStart,
                               const std::int64_t* column_samples,
                               int* mutex,
                               volatile Split<DataT>* splits,
                               ObjectiveT& objective,
                               const WorkloadInfo* workload_info,
                               dim3 histogram_grid,
                               dim3 split_grid,
                               const SharedMemoryConfig& split_smem_config,
                               cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
