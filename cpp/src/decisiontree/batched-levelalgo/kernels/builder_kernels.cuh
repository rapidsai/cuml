/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuml/common/utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <cub/cub.cuh>

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

/* Returns 'dataset' rounded up to a correctly-aligned pointer of type OutT* */
template <typename OutT, typename InT>
DI OutT* alignPointer(InT dataset)
{
  return reinterpret_cast<OutT*>(raft::alignTo(reinterpret_cast<size_t>(dataset), sizeof(OutT)));
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
void launchNodeSplitKernel(const IdxT max_depth,
                           const IdxT min_samples_leaf,
                           const IdxT min_samples_split,
                           const IdxT max_leaves,
                           const DataT min_impurity_decrease,
                           const Dataset<DataT, LabelT, IdxT>& dataset,
                           const NodeWorkItem* work_items,
                           const size_t work_items_size,
                           const Split<DataT, IdxT>* splits,
                           cudaStream_t builder_stream);

template <typename DatasetT, typename NodeT, typename ObjectiveT, typename DataT>
void launchLeafKernel(ObjectiveT objective,
                      DatasetT& dataset,
                      const NodeT* tree,
                      const InstanceRange* instance_ranges,
                      DataT* leaves,
                      int batch_size,
                      size_t smem_size,
                      cudaStream_t builder_stream);
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

template <typename IdxT>
struct CustomDifference {
  __device__ IdxT operator()(const IdxT& lhs, const IdxT& rhs)
  {
    if (lhs == rhs)
      return 0;
    else
      return 1;
  }
};

/**
 * @brief Generates 'k' unique samples of features from 'n' feature sample-space.
 *        Does this for each work-item (node), feeding a unique seed for each (treeid, nodeid
 * (=blockIdx.x), threadIdx.x). Method used is a random, parallel, sampling with replacement of
 * excess of 'k' samples (hence the name) and then eliminating the dupicates by ordering them. The
 * excess number of samples (=`n_parallel_samples`) is calculated such that after ordering there is
 * at least 'k' uniques.
 */
template <typename IdxT, int MAX_SAMPLES_PER_THREAD, int BLOCK_THREADS = 128>
CUML_KERNEL void excess_sample_with_replacement_kernel(
  IdxT* colids,
  const NodeWorkItem* work_items,
  size_t work_items_size,
  IdxT treeid,
  uint64_t seed,
  size_t n /* total cols to sample from*/,
  size_t k /* number of unique cols to sample */,
  int n_parallel_samples /* number of cols to sample with replacement */)
{
  if (blockIdx.x >= work_items_size) return;

  const uint32_t nodeid = work_items[blockIdx.x].idx;

  uint64_t subsequence(fnv1a32_basis);
  subsequence = fnv1a32(subsequence, uint32_t(threadIdx.x));
  subsequence = fnv1a32(subsequence, uint32_t(treeid));
  subsequence = fnv1a32(subsequence, uint32_t(nodeid));

  raft::random::PCGenerator gen(seed, subsequence, uint64_t(0));
  raft::random::UniformIntDistParams<IdxT, uint64_t> uniform_int_dist_params;

  uniform_int_dist_params.start = 0;
  uniform_int_dist_params.end   = n;
  uniform_int_dist_params.diff =
    uint64_t(uniform_int_dist_params.end - uniform_int_dist_params.start);

  IdxT n_uniques = 0;
  IdxT items[MAX_SAMPLES_PER_THREAD];
  IdxT col_indices[MAX_SAMPLES_PER_THREAD];
  IdxT mask[MAX_SAMPLES_PER_THREAD];
  // populate this
  for (int i = 0; i < MAX_SAMPLES_PER_THREAD; ++i)
    mask[i] = 0;

  do {
    // blocked arrangement
    for (int cta_sample_idx = MAX_SAMPLES_PER_THREAD * threadIdx.x, thread_local_sample_idx = 0;
         thread_local_sample_idx < MAX_SAMPLES_PER_THREAD;
         ++cta_sample_idx, ++thread_local_sample_idx) {
      // mask of the previous iteration, if exists, is re-used here
      // so previously generated unique random numbers are used.
      // newly generated random numbers may or may not duplicate the previously generated ones
      // but this ensures some forward progress in order to generate at least 'k' unique random
      // samples.
      if (mask[thread_local_sample_idx] == 0 and cta_sample_idx < n_parallel_samples)
        raft::random::custom_next(
          gen, &items[thread_local_sample_idx], uniform_int_dist_params, IdxT(0), IdxT(0));
      else if (mask[thread_local_sample_idx] ==
               0)  // indices that exceed `n_parallel_samples` will not generate
        items[thread_local_sample_idx] = n - 1;
      else
        continue;  // this case is for samples whose mask == 1 (saving previous iteration's random
                   // number generated)
    }

    // Specialize BlockRadixSort type for our thread block
    typedef cub::BlockRadixSort<IdxT, BLOCK_THREADS, MAX_SAMPLES_PER_THREAD> BlockRadixSortT;
    // BlockAdjacentDifference
    typedef cub::BlockAdjacentDifference<IdxT, BLOCK_THREADS> BlockAdjacentDifferenceT;
    // BlockScan
    typedef cub::BlockScan<IdxT, BLOCK_THREADS> BlockScanT;

    // Shared memory
    __shared__ union TempStorage {
      typename BlockRadixSortT::TempStorage sort;
      typename BlockAdjacentDifferenceT::TempStorage diff;
      typename BlockScanT::TempStorage scan;
    } temp_storage;

    // collectively sort items
    BlockRadixSortT(temp_storage.sort).Sort(items);

    __syncthreads();

    // compute the mask
    // compute the adjacent differences according to the functor
    // TODO: Replace deprecated 'FlagHeads' with 'SubtractLeft' when it is available
    BlockAdjacentDifferenceT(temp_storage.diff)
      .SubtractLeft(items, mask, CustomDifference<IdxT>(), mask[0]);

    __syncthreads();

    // do a scan on the mask to get the indices for gathering
    BlockScanT(temp_storage.scan).ExclusiveSum(mask, col_indices, n_uniques);

    __syncthreads();

  } while (n_uniques < k);

  // write the items[] of only the ones with mask[]=1 to col[offset + col_idx[]]
  IdxT col_offset = k * blockIdx.x;
  for (int i = 0; i < MAX_SAMPLES_PER_THREAD; ++i) {
    if (mask[i] and col_indices[i] < k) { colids[col_offset + col_indices[i]] = items[i]; }
  }
}

// algo L of the reservoir sampling algorithm
/**
 * @brief For each work item select 'k' features without replacement from 'n' features using algo-L.

* On exit each row of the colids array will contain k random integers from the [0..n-1] range.
*
* Each thread works on single row. The parameters work_items_size, treeid and seed are
* used to initialize a unique random seed for each work item.
*
* @param colids the generated random indices, size [work_items_size, k] row major layout
* @param work_items
* @param treeid
* @param seed
* @param n total cos to sample from
* @param k number of cols to sample
 * algorithm of reservoir sampling. wiki :
 * https://en.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm
 */
template <typename IdxT>
CUML_KERNEL void algo_L_sample_kernel(int* colids,
                                      const NodeWorkItem* work_items,
                                      size_t work_items_size,
                                      IdxT treeid,
                                      uint64_t seed,
                                      size_t n /* total cols to sample from*/,
                                      size_t k /* cols to sample */)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= work_items_size) return;
  const uint32_t nodeid = work_items[tid].idx;
  uint64_t subsequence  = (uint64_t(treeid) << 32) | uint64_t(nodeid);
  raft::random::PCGenerator gen(seed, subsequence, uint64_t(0));
  raft::random::UniformIntDistParams<IdxT, uint64_t> uniform_int_dist_params;
  uniform_int_dist_params.start = 0;
  uniform_int_dist_params.end   = k;
  uniform_int_dist_params.diff =
    uint64_t(uniform_int_dist_params.end - uniform_int_dist_params.start);
  float fp_uniform_val;
  IdxT int_uniform_val;
  // fp_uniform_val will have a random value between 0 and 1
  gen.next(fp_uniform_val);
  double W = raft::exp(raft::log(fp_uniform_val) / k);

  size_t col(0);
  // initially fill the reservoir array in increasing order of cols till k
  while (1) {
    colids[tid * k + col] = col;
    if (col == k - 1)
      break;
    else
      ++col;
  }
  // randomly sample from a geometric distribution
  while (col < n) {
    // fp_uniform_val will have a random value between 0 and 1
    gen.next(fp_uniform_val);
    col += static_cast<int>(raft::log(fp_uniform_val) / raft::log(1 - W)) + 1;
    if (col < n) {
      // int_uniform_val will now have a random value between 0...k
      raft::random::custom_next(gen, &int_uniform_val, uniform_int_dist_params, IdxT(0), IdxT(0));
      colids[tid * k + int_uniform_val] = col;  // the bad memory coalescing here is hidden
      // fp_uniform_val will have a random value between 0 and 1
      gen.next(fp_uniform_val);
      W *= raft::exp(raft::log(fp_uniform_val) / k);
    }
  }
}

template <typename IdxT>
CUML_KERNEL void adaptive_sample_kernel(int* colids,
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
void launchComputeSplitKernel(BinT* histograms,
                              IdxT n_bins,
                              IdxT max_depth,
                              IdxT min_samples_split,
                              IdxT max_leaves,
                              const Dataset<DataT, LabelT, IdxT>& dataset,
                              const Quantiles<DataT, IdxT>& quantiles,
                              const NodeWorkItem* work_items,
                              IdxT colStart,
                              const IdxT* colids,
                              int* done_count,
                              int* mutex,
                              volatile Split<DataT, IdxT>* splits,
                              ObjectiveT& objective,
                              IdxT treeid,
                              const WorkloadInfo<IdxT>* workload_info,
                              uint64_t seed,
                              dim3 grid,
                              size_t smem_size,
                              cudaStream_t builder_stream);

}  // namespace DT
}  // namespace ML
