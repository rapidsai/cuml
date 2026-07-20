/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder_kernels_impl.cuh"

namespace ML {
namespace DT {

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<float, int, int, TPB_DEFAULT>(
  const Dataset<float, int, int>& dataset,
  const NodeWorkItem* work_items,
  Split<float, int>* splits,
  const WorkloadInfo<int>* workload_info,
  size_t n_blocks_dimx,
  size_t n_work_items,
  int* partition_row_ids,
  cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<double, int, int, TPB_DEFAULT>(
  const Dataset<double, int, int>& dataset,
  const NodeWorkItem* work_items,
  Split<double, int>* splits,
  const WorkloadInfo<int>* workload_info,
  size_t n_blocks_dimx,
  size_t n_work_items,
  int* partition_row_ids,
  cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<float, float, int, TPB_DEFAULT>(
  const Dataset<float, float, int>& dataset,
  const NodeWorkItem* work_items,
  Split<float, int>* splits,
  const WorkloadInfo<int>* workload_info,
  size_t n_blocks_dimx,
  size_t n_work_items,
  int* partition_row_ids,
  cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<double, double, int, TPB_DEFAULT>(
  const Dataset<double, double, int>& dataset,
  const NodeWorkItem* work_items,
  Split<double, int>* splits,
  const WorkloadInfo<int>* workload_info,
  size_t n_blocks_dimx,
  size_t n_work_items,
  int* partition_row_ids,
  cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
