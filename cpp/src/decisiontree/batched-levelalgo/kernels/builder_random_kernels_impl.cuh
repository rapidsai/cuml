/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "builder_kernels.cuh"

#include <common/grid_sync.cuh>

#include <raft/util/cuda_utils.cuh>

#include <type_traits>

namespace ML {
namespace DT {

// TU-local: each kernel .cu includes exactly one impl header, so this never
// collides with the same-named constant in builder_kernels_impl.cuh.
static constexpr int TPB_DEFAULT = 128;

// Cross-block reduction reuses the pool primitive from the weighted
// RandomForest path; smem scales with n_classes alone (no per-bin histogram).
template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename ObjectiveT,
          typename BinT>
static __global__ void randomSplitKernel(BinT* histograms,
                                         BinT* pool,
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
                                         uint64_t seed,
                                         IdxT pool_slots)
{
  extern __shared__ char smem[];

  WorkloadInfo<IdxT> workload_info_cta = workload_info[blockIdx.x];
  IdxT nid                             = workload_info_cta.nodeid;
  IdxT large_nid                       = workload_info_cta.large_nodeid;
  const auto work_item                 = work_items[nid];
  auto range_start                     = work_item.instances.begin;
  auto range_len                       = work_item.instances.count;

  IdxT offset_blockid = workload_info_cta.offset_blockid;
  IdxT num_blocks     = workload_info_cta.num_blocks;

  IdxT col;
  if (dataset.n_sampled_cols == dataset.N) {
    col = colStart + blockIdx.y;
  } else {
    IdxT colIndex = colStart + blockIdx.y;
    col           = colids[nid * dataset.n_sampled_cols + colIndex];
  }

  int n_bins = quantiles.n_bins_array[col];
  // Constant feature (n_bins == 1): et_split_position's Lemire chain is
  // undefined at n_bins < 2. All blocks for this (node, feature) read the
  // same n_bins and return uniformly, so signalDone is never reached.
  if (n_bins < 2) { return; }
  IdxT per_side_cells = objective.NumClasses();
  auto* shared_left   = alignPointer<BinT>(smem);
  auto* shared_parent = alignPointer<BinT>(shared_left + per_side_cells);
  auto* shared_done   = alignPointer<int>(shared_parent + per_side_cells);

  for (IdxT j = threadIdx.x; j < per_side_cells; j += blockDim.x) {
    shared_left[j]   = BinT();
    shared_parent[j] = BinT();
  }
  if (threadIdx.x == 0) *shared_done = 0;
  __syncthreads();

  IdxT split_pos =
    et_split_position<IdxT>(seed, uint64_t(treeid), uint64_t(work_item.idx), uint64_t(col), n_bins);
  DataT threshold = quantiles.quantiles_array[max_n_bins * col + split_pos];

  IdxT stride            = blockDim.x * num_blocks;
  IdxT tid               = threadIdx.x + offset_blockid * blockDim.x;
  std::size_t col_offset = std::size_t(col) * dataset.M;
  auto end               = range_start + range_len;

  for (auto i = range_start + tid; i < end; i += stride) {
    auto row   = dataset.row_ids[i];
    auto data  = dataset.data[row + col_offset];
    auto label = dataset.labels[row];
    if constexpr (std::is_same_v<BinT, WeightedCountBin> ||
                  std::is_same_v<BinT, WeightedAggregateBin>) {
      auto weight = dataset.sample_weight ? dataset.sample_weight[row] : DataT(1.0);
      BinT::IncrementHistogram(shared_parent, 1, 0, label, weight);
      if (data <= threshold) { BinT::IncrementHistogram(shared_left, 1, 0, label, weight); }
    } else {
      BinT::IncrementHistogram(shared_parent, 1, 0, label);
      if (data <= threshold) { BinT::IncrementHistogram(shared_left, 1, 0, label); }
    }
  }

  __syncthreads();

  if (num_blocks > 1) {
    if constexpr (std::is_same_v<BinT, CountBin> || std::is_same_v<BinT, AggregateBin>) {
      // AggregateBin double-atomicAdd inherits RF's R7 trade-off (#8093).
      auto offset = ((large_nid * gridDim.y) + blockIdx.y) * IdxT(2) * per_side_cells;
      for (IdxT j = threadIdx.x; j < per_side_cells; j += blockDim.x) {
        BinT::AtomicAdd(histograms + offset + j, shared_left[j]);
        BinT::AtomicAdd(histograms + offset + per_side_cells + j, shared_parent[j]);
      }
      __threadfence();
      __syncthreads();
      bool last = MLCommon::signalDone(
        done_count + nid * gridDim.y + blockIdx.y, num_blocks, offset_blockid == 0, shared_done);
      if (!last) return;
      for (IdxT j = threadIdx.x; j < per_side_cells; j += blockDim.x) {
        shared_left[j]   = histograms[offset + j];
        shared_parent[j] = histograms[offset + per_side_cells + j];
      }
      __syncthreads();
    } else {
      // #8132's pool primitive; per-slot footprint shrinks max_n_bins -> 1.
      auto pool_slot_stride = IdxT(2) * per_side_cells;
      auto slot             = offset_blockid % pool_slots;
      auto pool_offset = ((large_nid * gridDim.y) + blockIdx.y) * pool_slots * pool_slot_stride +
                         slot * pool_slot_stride;
      for (IdxT j = threadIdx.x; j < per_side_cells; j += blockDim.x) {
        BinT::AtomicAdd(pool + pool_offset + j, shared_left[j]);
        BinT::AtomicAdd(pool + pool_offset + per_side_cells + j, shared_parent[j]);
      }
      __threadfence();
      __syncthreads();
      bool last = MLCommon::signalDone(
        done_count + nid * gridDim.y + blockIdx.y, num_blocks, offset_blockid == 0, shared_done);
      if (!last) return;
      auto pool_base = ((large_nid * gridDim.y) + blockIdx.y) * pool_slots * pool_slot_stride;
      int n_slots =
        num_blocks < pool_slots ? static_cast<int>(num_blocks) : static_cast<int>(pool_slots);
      for (IdxT j = threadIdx.x; j < per_side_cells; j += blockDim.x) {
        BinT merged_l{}, merged_p{};
        for (int s = 0; s < n_slots; ++s) {
          merged_l += pool[pool_base + s * pool_slot_stride + j];
          merged_p += pool[pool_base + s * pool_slot_stride + per_side_cells + j];
        }
        shared_left[j]   = merged_l;
        shared_parent[j] = merged_p;
      }
      __syncthreads();
    }
  }

  // Only thread 0 scores; other threads carry default-init Split (worst gain).
  Split<DataT, IdxT> sp;
  if (threadIdx.x == 0) {
    DataT gain = objective.GainFromSideStats(shared_left, shared_parent);
    IdxT nLeft = 0;
    if constexpr (std::is_same_v<BinT, CountBin>) {
      for (IdxT j = 0; j < per_side_cells; ++j)
        nLeft += shared_left[j].x;
    } else if constexpr (std::is_same_v<BinT, WeightedCountBin>) {
      for (IdxT j = 0; j < per_side_cells; ++j)
        nLeft += shared_left[j].count;
    } else {
      nLeft = shared_left[0].count;
    }
    sp.update({threshold, col, gain, nLeft});
  }

  __syncthreads();

  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename ObjectiveT,
          typename BinT>
void launchRandomSplitKernel(BinT* histograms,
                             BinT* pool,
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
                             IdxT pool_slots,
                             dim3 grid,
                             size_t smem_size,
                             cudaStream_t builder_stream)
{
  randomSplitKernel<DataT, LabelT, IdxT, TPB, ObjectiveT, BinT>
    <<<grid, TPB, smem_size, builder_stream>>>(histograms,
                                               pool,
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
                                               seed,
                                               pool_slots);
}

// One explicit instantiation per random_*-{float,double}.cu translation unit
// that includes this header.
template void launchRandomSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT, _ObjectiveT, _BinT>(
  _BinT* histograms,
  _BinT* pool,
  _IdxT max_n_bins,
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
  _IdxT pool_slots,
  dim3 grid,
  size_t smem_size,
  cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
