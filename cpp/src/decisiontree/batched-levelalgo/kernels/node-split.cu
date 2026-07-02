/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder_kernels_impl.cuh"

namespace ML {
namespace DT {

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<float, int, TPB_DEFAULT>(const int min_samples_leaf,
                                                             const float min_impurity_decrease,
                                                             const Dataset<float, int>& dataset,
                                                             const NodeWorkItem* work_items,
                                                             const Split<float>* splits,
                                                             const WorkloadInfo* workload_info,
                                                             size_t n_blocks_dimx,
                                                             int* partition_row_ids,
                                                             cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<double, int, TPB_DEFAULT>(const int min_samples_leaf,
                                                              const double min_impurity_decrease,
                                                              const Dataset<double, int>& dataset,
                                                              const NodeWorkItem* work_items,
                                                              const Split<double>* splits,
                                                              const WorkloadInfo* workload_info,
                                                              size_t n_blocks_dimx,
                                                              int* partition_row_ids,
                                                              cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<float, float, TPB_DEFAULT>(const int min_samples_leaf,
                                                               const float min_impurity_decrease,
                                                               const Dataset<float, float>& dataset,
                                                               const NodeWorkItem* work_items,
                                                               const Split<float>* splits,
                                                               const WorkloadInfo* workload_info,
                                                               size_t n_blocks_dimx,
                                                               int* partition_row_ids,
                                                               cudaStream_t builder_stream);

// Explicit instantiations are split across separate .cu files to increase compilation parallelism.
template void launchNodeSplitKernel<double, double, TPB_DEFAULT>(
  const int min_samples_leaf,
  const double min_impurity_decrease,
  const Dataset<double, double>& dataset,
  const NodeWorkItem* work_items,
  const Split<double>* splits,
  const WorkloadInfo* workload_info,
  size_t n_blocks_dimx,
  int* partition_row_ids,
  cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
