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

#include <raft/linalg/unary_op.cuh>

#include <cuda/iterator>
#include <cuda/std/random>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

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
  int nodeid;          // Node in the batch on which the threadblock needs to work
  int offset_blockid;  // Offset threadblock id among all the blocks that are
                       // working on this node
  int num_blocks;      // Total number of blocks that are working on the node
};

template <typename SplitT, typename IdxT>
HDI bool SplitPartitionNotValid(const SplitT& split, IdxT, std::size_t)
{
  return split.colid == -1;
}

template <typename SplitT, typename DataT>
HDI bool SplitNotValid(const SplitT& split, DataT min_impurity_decrease)
{
  return split.colid == -1 || split.best_metric_val <= min_impurity_decrease;
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
void launchNodeSplitKernel(const DataT min_impurity_decrease,
                           const Dataset<DataT, LabelT, IdxT>& dataset,
                           const NodeWorkItem* work_items,
                           Split<DataT>* splits,
                           const WorkloadInfo* workload_info,
                           size_t n_blocks_dimx,
                           IdxT* partition_row_ids,
                           cudaStream_t builder_stream);

template <typename DatasetT, typename NodeT, typename ObjectiveT>
void launchLeafHistogramKernel(ObjectiveT objective,
                               DatasetT& dataset,
                               const NodeT* tree,
                               const InstanceRange* instance_ranges,
                               typename ObjectiveT::BinT* leaf_histograms,
                               int batch_size,
                               size_t smem_size,
                               cudaStream_t builder_stream);

template <typename NodeT, typename ObjectiveT, typename DataT>
void launchFinalizeLeafKernel(ObjectiveT objective,
                              const NodeT* tree,
                              const typename ObjectiveT::BinT* leaf_histograms,
                              DataT* leaves,
                              int batch_size,
                              int num_outputs,
                              cudaStream_t builder_stream);
// Returns the lowest index in `array` whose value is greater or equal to `element`.
// Values outside the quantile range are clamped to the edge bins: values below the
// first quantile return 0, and values above the last quantile return len - 1.
template <typename DataT, typename IdxT>
HDI IdxT lower_bound(DataT* array, IdxT len, DataT element)
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
                                       cudaStream_t builder_stream);

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
                               cudaStream_t builder_stream);

template <typename BinT>
inline constexpr bool has_label_sum_v =
  std::is_same_v<BinT, RegressionBin> || std::is_same_v<BinT, WeightedRegressionBin>;

template <typename BinT>
inline constexpr bool has_weight_v =
  std::is_same_v<BinT, WeightedClassificationBin> || std::is_same_v<BinT, WeightedRegressionBin>;

template <typename BinT>
inline void packHistograms(const BinT* in,
                           double* label_sums,
                           std::uint64_t* counts,
                           double* weights,
                           std::size_t len,
                           cudaStream_t stream)
{
  if constexpr (has_label_sum_v<BinT>) {
    auto label_sum_op = [in] __device__(double* out, std::size_t i) { *out = in[i].LabelSum(); };
    raft::linalg::writeOnlyUnaryOp<double, decltype(label_sum_op), std::size_t, 256>(
      label_sums, len, label_sum_op, stream);
  }

  auto count_op = [in] __device__(std::uint64_t* out, std::size_t i) { *out = in[i].Count(); };
  raft::linalg::writeOnlyUnaryOp<std::uint64_t, decltype(count_op), std::size_t, 256>(
    counts, len, count_op, stream);

  if constexpr (has_weight_v<BinT>) {
    auto weight_op = [in] __device__(double* out, std::size_t i) { *out = in[i].Weight(); };
    raft::linalg::writeOnlyUnaryOp<double, decltype(weight_op), std::size_t, 256>(
      weights, len, weight_op, stream);
  }
}

inline void unpackHistograms(const double*,
                             const std::uint64_t* counts,
                             const double*,
                             ClassificationBin* out,
                             std::size_t len,
                             cudaStream_t stream)
{
  auto op = [counts] __device__(ClassificationBin * out, std::size_t i) { out->count = counts[i]; };
  raft::linalg::writeOnlyUnaryOp<ClassificationBin, decltype(op), std::size_t, 256>(
    out, len, op, stream);
}

inline void unpackHistograms(const double*,
                             const std::uint64_t* counts,
                             const double* weights,
                             WeightedClassificationBin* out,
                             std::size_t len,
                             cudaStream_t stream)
{
  auto op = [counts, weights] __device__(WeightedClassificationBin * out, std::size_t i) {
    out->count  = counts[i];
    out->weight = weights[i];
  };
  raft::linalg::writeOnlyUnaryOp<WeightedClassificationBin, decltype(op), std::size_t, 256>(
    out, len, op, stream);
}

inline void unpackHistograms(const double* label_sums,
                             const std::uint64_t* counts,
                             const double*,
                             RegressionBin* out,
                             std::size_t len,
                             cudaStream_t stream)
{
  auto op = [label_sums, counts] __device__(RegressionBin * out, std::size_t i) {
    out->label_sum = label_sums[i];
    out->count     = counts[i];
  };
  raft::linalg::writeOnlyUnaryOp<RegressionBin, decltype(op), std::size_t, 256>(
    out, len, op, stream);
}

inline void unpackHistograms(const double* label_sums,
                             const std::uint64_t* counts,
                             const double* weights,
                             WeightedRegressionBin* out,
                             std::size_t len,
                             cudaStream_t stream)
{
  auto op = [label_sums, counts, weights] __device__(WeightedRegressionBin * out, std::size_t i) {
    out->label_sum = label_sums[i];
    out->count     = counts[i];
    out->weight    = weights[i];
  };
  raft::linalg::writeOnlyUnaryOp<WeightedRegressionBin, decltype(op), std::size_t, 256>(
    out, len, op, stream);
}

}  // namespace DT
}  // namespace ML
