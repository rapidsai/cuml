/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/binary_search.h>
#include <common/grid_sync.cuh>
#include <cstdio>
#include <cub/cub.cuh>
#include <raft/cuda_utils.cuh>
#include <cuml/tree/flatnode.h>
#include "builder_kernels.cuh"

namespace ML {
namespace DT {
template
__global__ void nodeSplitKernel<float, float, int, MSEObjectiveFunction<float, float, int>, TPB_DEFAULT>(int max_depth,
                                int min_samples_leaf,
                                int min_samples_split,
                                int max_leaves,
                                float min_impurity_decrease,
                                Input<float, float, int> input,
                                NodeWorkItem* work_items,
                                const Split<float, int>* splits);
template __global__ void leafKernel< Input<float, float, int>,  SparseTreeNode<float, float, int>,  MSEObjectiveFunction<float, float, int>,  float>
(MSEObjectiveFunction<float, float, int> objective,
                           Input<float, float, int> input,
                           const SparseTreeNode<float, float, int>* tree,
                           const InstanceRange* instance_ranges,
                           float* leaves);
template
__global__ void computeSplitKernel< float,
           float,
           int,
           TPB_DEFAULT,
           MSEObjectiveFunction<float, float, int>,
           AggregateBin>(AggregateBin* hist,
                                   int nbins,
                                   int max_depth,
                                   int min_samples_split,
                                   int max_leaves,
                                   Input<float, float, int> input,
                                   const NodeWorkItem* work_items,
                                   int colStart,
                                   int* done_count,
                                   int* mutex,
                                   volatile Split<float, int>* splits,
                                   MSEObjectiveFunction<float, float, int> objective,
                                   int treeid,
                                   const WorkloadInfo<int>* workload_info,
                                   uint64_t seed);
}  // namespace DT
}  // namespace ML
