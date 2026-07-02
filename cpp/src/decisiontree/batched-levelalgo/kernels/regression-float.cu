/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder_kernels_impl.cuh"

#include <cuml/tree/flatnode.h>

namespace ML {
namespace DT {
using DataT      = float;
using LabelT     = float;
using ObjectiveT = RegressionObjectiveFunction<DataT, LabelT>;
using BinT       = typename ObjectiveT::BinT;
using DatasetT   = Dataset<DataT, LabelT>;
using NodeT      = SparseTreeNode<DataT, LabelT>;

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
template void launchComputeSplitKernel<DataT, LabelT, TPB_DEFAULT, ObjectiveT>(
  BinT* histograms,
  int n_bins,
  int min_samples_split,
  int max_leaves,
  const DatasetT& dataset,
  const Quantiles<DataT>& quantiles,
  const NodeWorkItem* work_items,
  int colStart,
  const int* column_samples,
  int* done_count,
  int* mutex,
  volatile Split<DataT>* splits,
  ObjectiveT& objective,
  int treeid,
  const WorkloadInfo* workload_info,
  uint64_t seed,
  dim3 grid,
  size_t smem_size,
  cudaStream_t builder_stream);
}  // namespace DT
}  // namespace ML
