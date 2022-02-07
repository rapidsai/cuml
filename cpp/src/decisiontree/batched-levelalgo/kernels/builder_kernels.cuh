/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#pragma once

#include "../bins.cuh"
#include "../objectives.cuh"
#include "../quantiles.h"

namespace ML {
namespace DT {

// The range of instances belonging to a particular node
// This structure refers to a range in the device array dataset.row_ids
struct InstanceRange {
  std::size_t begin;
  std::size_t count;
};

struct NodeWorkItem {
  size_t idx;  // Index of the work item in the tree
  int depth;
  InstanceRange instances;
};

/**
 * This struct has information about workload of a single threadblock of
 * computeSplit kernels of classification and regression
 */
template <typename IdxT>
struct WorkloadInfo {
  IdxT nodeid;        // Node in the batch on which the threadblock needs to work
  IdxT large_nodeid;  // counts only large nodes (nodes that require more than one block along x-dim
                      // for histogram calculation)
  IdxT offset_blockid;  // Offset threadblock id among all the blocks that are
                        // working on this node
  IdxT num_blocks;      // Total number of blocks that are working on the node
};

template <typename SplitT, typename DataT, typename IdxT>
HDI bool SplitNotValid(const SplitT& split,
                       DataT min_impurity_decrease,
                       IdxT min_samples_leaf,
                       std::size_t num_rows)
{
  return split.best_metric_val <= min_impurity_decrease || split.nLeft < min_samples_leaf ||
         (IdxT(num_rows) - split.nLeft) < min_samples_leaf;
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
__global__ void nodeSplitKernel(const IdxT max_depth,
                                const IdxT min_samples_leaf,
                                const IdxT min_samples_split,
                                const IdxT max_leaves,
                                const DataT min_impurity_decrease,
                                const Dataset<DataT, LabelT, IdxT> dataset,
                                const NodeWorkItem* work_items,
                                const Split<DataT, IdxT>* splits);

template <typename DatasetT, typename NodeT, typename ObjectiveT, typename DataT>
__global__ void leafKernel(ObjectiveT objective,
                           DatasetT dataset,
                           const NodeT* tree,
                           const InstanceRange* instance_ranges,
                           DataT* leaves);
// 32-bit FNV1a hash
// Reference: http://www.isthe.com/chongo/tech/comp/fnv/index.html
const uint32_t fnv1a32_prime = uint32_t(16777619);
const uint32_t fnv1a32_basis = uint32_t(2166136261);
HDI uint32_t fnv1a32(uint32_t hash, uint32_t txt)
{
  hash ^= (txt >> 0) & 0xFF;
  hash *= fnv1a32_prime;
  hash ^= (txt >> 8) & 0xFF;
  hash *= fnv1a32_prime;
  hash ^= (txt >> 16) & 0xFF;
  hash *= fnv1a32_prime;
  hash ^= (txt >> 24) & 0xFF;
  hash *= fnv1a32_prime;
  return hash;
}

// returns the lowest index in `array` whose value is greater or equal to `element`
template <typename DataT, typename IdxT>
HDI IdxT lower_bound(DataT* array, IdxT len, DataT element)
{
  IdxT start = 0;
  IdxT end   = len - 1;
  IdxT mid;
  while (start < end) {
    mid = (start + end) / 2;
    if (array[mid] < element) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename DataT,
          typename LabelT,
          typename IdxT,
          int TPB,
          typename ObjectiveT,
          typename BinT>
__global__ void computeSplitKernel(BinT* histograms,
                                   IdxT n_bins,
                                   IdxT max_depth,
                                   IdxT min_samples_split,
                                   IdxT max_leaves,
                                   const Dataset<DataT, LabelT, IdxT> dataset,
                                   const Quantiles<DataT, IdxT> quantiles,
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
