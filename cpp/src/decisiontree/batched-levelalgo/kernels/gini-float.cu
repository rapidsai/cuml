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
  using DataT      = float;
  using LabelT     = int;
  using IdxT       = int;
  using ObjectiveT = GiniObjectiveFunction<DataT, LabelT, IdxT>;
  using BinT       = CountBin;
  using InputT     = Input<DataT, LabelT, IdxT>;
  using NodeT      = SparseTreeNode<DataT, LabelT, IdxT>;
  // "almost" instantiation templates to avoid code-duplication
  template
  __global__ void nodeSplitKernel< DataT,  LabelT,  IdxT,  TPB_DEFAULT>(IdxT max_depth,
                                  IdxT min_samples_leaf,
                                  IdxT min_samples_split,
                                  IdxT max_leaves,
                                  DataT min_impurity_decrease,
                                  Input<DataT, LabelT, IdxT> input,
                                  NodeWorkItem* work_items,
                                  const Split<DataT, IdxT>* splits);

  template
  __global__ void leafKernel< InputT,  NodeT,  ObjectiveT,  DataT>(ObjectiveT objective,
                            InputT input,
                            const NodeT* tree,
                            const InstanceRange* instance_ranges,
                            DataT* leaves);
  template
  __global__ void computeSplitKernel< DataT,
            LabelT,
            IdxT,
            TPB_DEFAULT,
            ObjectiveT,
            BinT>(BinT* hist,
                                    IdxT nbins,
                                    IdxT max_depth,
                                    IdxT min_samples_split,
                                    IdxT max_leaves,
                                    Input<DataT, LabelT, IdxT> input,
                                    const NodeWorkItem* work_items,
                                    IdxT colStart,
                                    int* done_count,
                                    int* mutex,
                                    volatile Split<DataT, IdxT>* splits,
                                    ObjectiveT objective,
                                    IdxT treeid,
                                    const WorkloadInfo<IdxT>* workload_info,
                                    uint64_t seed);
}  // namespace DT
}  // namespace ML
