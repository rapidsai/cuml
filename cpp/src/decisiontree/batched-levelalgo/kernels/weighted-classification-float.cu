/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
template void launchBuildHistogramsKernel<DataT, LabelT, IdxT, TPB_DEFAULT, ObjectiveT>(
  BinT* histograms,
  IdxT n_bins,
  const DatasetT& dataset,
  const Quantiles<DataT, IdxT>& quantiles,
  const NodeWorkItem* work_items,
  IdxT colStart,
  const IdxT* column_samples,
  ObjectiveT& objective,
  const WorkloadInfo<IdxT>* workload_info,
  dim3 histogram_grid,
  const SharedMemoryConfig& split_smem_config,
  cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchFindBestSplitsKernel<DataT, LabelT, IdxT, TPB_DEFAULT, ObjectiveT>(
  BinT* histograms,
  IdxT n_bins,
  const DatasetT& dataset,
  const Quantiles<DataT, IdxT>& quantiles,
  IdxT colStart,
  const IdxT* column_samples,
  int* mutex,
  volatile Split<DataT, IdxT>* splits,
  ObjectiveT& objective,
  dim3 split_grid,
  cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
