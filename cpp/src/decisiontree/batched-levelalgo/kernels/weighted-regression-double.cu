/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder_kernels_impl.cuh"

#include <cuml/tree/flatnode.h>

namespace ML {
namespace DT {
using DataT      = double;
using LabelT     = double;
using ObjectiveT = RegressionObjectiveFunction<DataT, LabelT, std::int64_t, true>;
using BinT       = typename ObjectiveT::BinT;
using DatasetT   = Dataset<DataT, LabelT, std::int64_t>;
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
template void launchComputeSplitKernels<DataT, LabelT, std::int64_t, TPB_DEFAULT, ObjectiveT>(
  BinT* histograms,
  std::int64_t n_bins,
  const DatasetT& dataset,
  const Quantiles<DataT>& quantiles,
  const NodeWorkItem* work_items,
  std::int64_t colStart,
  const std::int64_t* column_samples,
  int* mutex,
  volatile Split<DataT, std::int64_t>* splits,
  ObjectiveT& objective,
  const WorkloadInfo<std::int64_t>* workload_info,
  dim3 histogram_grid,
  dim3 split_grid,
  const SharedMemoryConfig& split_smem_config,
  cudaStream_t builder_stream);
}  // namespace DT
}  // namespace ML
