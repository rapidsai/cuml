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

#include <raft/random/rng.hpp>

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

/**
 * @brief For a given values of (treeid, nodeid, seed), this function generates
 *        a unique permutation of [0, N - 1] values and returns 'k'th entry in
 *        from the permutation.
 * @return The 'k'th value from the permutation
 * @note This function does not allocated any temporary buffer, all the
 *       necessary values are recomputed.
 */
template <typename IdxT>
DI IdxT select(IdxT k, IdxT treeid, uint32_t nodeid, uint64_t seed, IdxT N)
{
  __shared__ int blksum;
  uint32_t pivot_hash;
  int cnt = 0;

  if (threadIdx.x == 0) { blksum = 0; }
  // Compute hash for the 'k'th index and use it as pivote for sorting
  pivot_hash = fnv1a32_basis;
  pivot_hash = fnv1a32(pivot_hash, uint32_t(k));
  pivot_hash = fnv1a32(pivot_hash, uint32_t(treeid));
  pivot_hash = fnv1a32(pivot_hash, uint32_t(nodeid));
  pivot_hash = fnv1a32(pivot_hash, uint32_t(seed >> 32));
  pivot_hash = fnv1a32(pivot_hash, uint32_t(seed));

  // Compute hash for rest of the indices and count instances where i_hash is
  // less than pivot_hash
  uint32_t i_hash;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    if (i == k) continue;  // Skip since k is the pivote index
    i_hash = fnv1a32_basis;
    i_hash = fnv1a32(i_hash, uint32_t(i));
    i_hash = fnv1a32(i_hash, uint32_t(treeid));
    i_hash = fnv1a32(i_hash, uint32_t(nodeid));
    i_hash = fnv1a32(i_hash, uint32_t(seed >> 32));
    i_hash = fnv1a32(i_hash, uint32_t(seed));

    if (i_hash < pivot_hash)
      cnt++;
    else if (i_hash == pivot_hash && i < k)
      cnt++;
  }
  __syncthreads();
  if (cnt > 0) atomicAdd(&blksum, cnt);
  __syncthreads();
  return blksum;
}

template <typename IdxT>
__global__ void select_kernel(
  IdxT* colids, const NodeWorkItem* work_items, IdxT treeid, uint64_t seed, IdxT N)
{
  const uint32_t nodeid = work_items[blockIdx.x].idx;

  int blksum = select(IdxT(blockIdx.y), treeid, nodeid, seed, N);
  if (threadIdx.x == 0) { colids[blockIdx.x * gridDim.y + blockIdx.y] = blksum; }
}

// __device__ uint32_t static kiss99(uint32_t& z, uint32_t& w, uint32_t& jsr, uint32_t& jcong)
// {
//   uint32_t MWC;
//   z   = 36969 * (z & 65535) + (z >> 16);
//   w   = 18000 * (w & 65535) + (w >> 16);
//   MWC = ((z << 16) + w);
//   jsr ^= (jsr << 17);
//   jsr ^= (jsr >> 13);
//   jsr ^= (jsr << 5);
//   jcong = 69069 * jcong + 1234567;
//   return ((MWC ^ jcong) + jsr);
// }

template <typename IdxT>
__global__ void adaptive_sample_kernel(int* colids,
                                       const NodeWorkItem* work_items,
                                       size_t work_items_size,
                                       IdxT treeid,
                                       uint64_t seed,
                                       int N,
                                       int M)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= work_items_size) return;
  const uint32_t nodeid = work_items[tid].idx;

  uint64_t subsequence = (uint64_t(treeid) << 32) | uint64_t(nodeid);
  raft::random::PCGenerator gen(seed, subsequence, uint64_t(0));

  int selected_count = 0;
  for (int i = 0; i < N; i++) {
    uint32_t toss = 0;
    gen.next(toss);
    uint64_t lhs = uint64_t(M - selected_count);
    lhs <<= 32;
    uint64_t rhs = uint64_t(toss) * (N - i);
    if (lhs > rhs) {
      colids[tid * M + selected_count] = i;
      selected_count++;
      if (selected_count == M) break;
    }
  }
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
                                   const IdxT* colids,
                                   int* done_count,
                                   int* mutex,
                                   volatile Split<DataT, IdxT>* splits,
                                   ObjectiveT objective,
                                   IdxT treeid,
                                   const WorkloadInfo<IdxT>* workload_info,
                                   uint64_t seed);

}  // namespace DT
}  // namespace ML
