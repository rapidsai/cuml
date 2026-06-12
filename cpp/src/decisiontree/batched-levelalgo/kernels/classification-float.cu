/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../bins.cuh"
#include "../objectives.cuh"

#include <cuml/tree/flatnode.h>

namespace ML {
namespace DT {
using _DataT      = float;
using _LabelT     = int;
using _IdxT       = int;
using _ObjectiveT = ClassificationObjectiveFunction<_DataT, _LabelT, _IdxT>;
using _BinT       = ClassificationBin;
using _DatasetT   = Dataset<_DataT, _LabelT, _IdxT>;
using _NodeT      = SparseTreeNode<_DataT, _LabelT, _IdxT>;
}  // namespace DT
}  // namespace ML

#include "builder_kernels_impl.cuh"

namespace ML {
namespace DT {
using _WeightedObjectiveT = ClassificationObjectiveFunction<_DataT, _LabelT, _IdxT, true>;
using _WeightedBinT       = typename _WeightedObjectiveT::BinT;
template void launchLeafKernel<_DatasetT, _NodeT, _WeightedObjectiveT, _DataT>(
  _WeightedObjectiveT objective,
  _DatasetT& dataset,
  const _NodeT* tree,
  const InstanceRange* instance_ranges,
  _DataT* leaves,
  int batch_size,
  size_t smem_size,
  cudaStream_t builder_stream);

template void
launchComputeSplitKernel<_DataT, _LabelT, _IdxT, TPB_DEFAULT, _WeightedObjectiveT>(
  _WeightedBinT* histograms,
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
  _WeightedObjectiveT& objective,
  _IdxT treeid,
  const WorkloadInfo<_IdxT>* workload_info,
  uint64_t seed,
  dim3 grid,
  size_t smem_size,
  cudaStream_t builder_stream);
}  // namespace DT
}  // namespace ML
