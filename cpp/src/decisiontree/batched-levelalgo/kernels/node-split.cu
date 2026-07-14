/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder_kernels_impl.cuh"

namespace ML {
namespace DT {

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<float, int, std::int64_t, TPB_DEFAULT>(
  const std::int64_t min_samples_leaf,
  const float min_impurity_decrease,
  const Dataset<float, int, std::int64_t>& dataset,
  const NodeWorkItem* work_items,
  const Split<float, std::int64_t>* splits,
  const WorkloadInfo<std::int64_t>* workload_info,
  size_t n_blocks_dimx,
  std::int64_t* partition_row_ids,
  cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<double, int, std::int64_t, TPB_DEFAULT>(
  const std::int64_t min_samples_leaf,
  const double min_impurity_decrease,
  const Dataset<double, int, std::int64_t>& dataset,
  const NodeWorkItem* work_items,
  const Split<double, std::int64_t>* splits,
  const WorkloadInfo<std::int64_t>* workload_info,
  size_t n_blocks_dimx,
  std::int64_t* partition_row_ids,
  cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<float, float, std::int64_t, TPB_DEFAULT>(
  const std::int64_t min_samples_leaf,
  const float min_impurity_decrease,
  const Dataset<float, float, std::int64_t>& dataset,
  const NodeWorkItem* work_items,
  const Split<float, std::int64_t>* splits,
  const WorkloadInfo<std::int64_t>* workload_info,
  size_t n_blocks_dimx,
  std::int64_t* partition_row_ids,
  cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<double, double, std::int64_t, TPB_DEFAULT>(
  const std::int64_t min_samples_leaf,
  const double min_impurity_decrease,
  const Dataset<double, double, std::int64_t>& dataset,
  const NodeWorkItem* work_items,
  const Split<double, std::int64_t>* splits,
  const WorkloadInfo<std::int64_t>* workload_info,
  size_t n_blocks_dimx,
  std::int64_t* partition_row_ids,
  cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
