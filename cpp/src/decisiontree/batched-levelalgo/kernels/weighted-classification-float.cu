/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder_kernels_impl.cuh"

#include <cuml/tree/flatnode.h>

namespace ML {
namespace DT {
using DataT      = float;
using LabelT     = int;
using IdxT       = int;
using ObjectiveT = ClassificationObjectiveFunction<DataT, LabelT, IdxT, true>;
using BinT       = typename ObjectiveT::BinT;
using DatasetT   = Dataset<DataT, LabelT, IdxT>;
using NodeT      = SparseTreeNode<DataT, LabelT, IdxT>;

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchLeafKernel<DatasetT, NodeT, ObjectiveT, DataT>(
  ObjectiveT objective,
  DatasetT& dataset,
  const NodeT* tree,
  const InstanceRange* instance_ranges,
  DataT* leaves,
  int batch_size,
  size_t smem_size,
  cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchComputeSplitKernel<DataT, LabelT, IdxT, TPB_DEFAULT, ObjectiveT>(
  BinT* histograms,
  IdxT n_bins,
  IdxT min_samples_split,
  IdxT max_leaves,
  const DatasetT& dataset,
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
