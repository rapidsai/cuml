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

#pragma once

#include <cuml/tree/algo_helper.h>
#include <thrust/binary_search.h>
#include <common/grid_sync.cuh>
#include <cub/cub.cuh>
#include <raft/cuda_utils.cuh>
#include "input.cuh"
#include "metrics.cuh"
#include "node.cuh"
#include "split.cuh"

namespace ML {
namespace DecisionTree {

/**
 * This struct has information about workload of a single threadblock of
 * computeSplit kernels of classification and regression
 */
template <typename IdxT>
struct WorkloadInfo {
  IdxT nodeid;  // Node in the batch on which the threadblock needs to work
  IdxT offset_blockid;  // Offset threadblock id among all the blocks that are
                        // working on this node
  IdxT num_blocks;      // Total number of blocks that are working on the node
};

/**
 * @brief Decide whether the current node is to be declared as a leaf entirely
 *        based on the input hyper-params set by the user
 *
 * @param[in] myDepth           depth of this node
 * @param[in] max_depth maximum possible tree depth
 * @param[in] min_samples_split min number of samples needed to split an
 *                              internal node
 * @param[in] max_leaves        max leaf nodes per tree (it's a soft constraint)
 * @param[in] n_leaves          number of leaves in the tree already
 * @param[in] nSamples          number of samples belonging to this node
 *
 * @return true if the current node is to be declared as a leaf, else false
 */
template <typename DataT, typename IdxT>
HDI bool leafBasedOnParams(IdxT myDepth, IdxT max_depth, IdxT min_samples_split,
                           IdxT max_leaves, const IdxT* n_leaves,
                           IdxT nSamples) {
  if (myDepth >= max_depth) return true;
  if (nSamples < min_samples_split) return true;
  if (max_leaves != -1) {
    if (*n_leaves >= max_leaves) return true;
  }
  return false;
}

/**
 * @brief Partition the samples to left/right nodes based on the best split
 * @return the position of the left child node in the nodes list. However, this
 *         value is valid only for threadIdx.x == 0.
 * @note this should be called by only one block from all participating blocks
 *       'smem' should be atleast of size `sizeof(IdxT) * TPB * 2`
 */
template <typename DataT, typename LabelT, typename IdxT, int TPB>
DI void partitionSamples(const Input<DataT, LabelT, IdxT>& input,
                         const Split<DataT, IdxT>* splits,
                         volatile Node<DataT, LabelT, IdxT>* curr_nodes,
                         volatile Node<DataT, LabelT, IdxT>* next_nodes,
                         IdxT* n_nodes, IdxT* n_depth, IdxT total_nodes,
                         char* smem) {
  typedef cub::BlockScan<int, TPB> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp1, temp2;
  volatile auto* rowids = reinterpret_cast<volatile IdxT*>(input.rowids);
  // for compaction
  size_t smemSize = sizeof(IdxT) * TPB;
  auto* lcomp = reinterpret_cast<IdxT*>(smem);
  auto* rcomp = reinterpret_cast<IdxT*>(smem + smemSize);
  auto nid = blockIdx.x;
  auto split = splits[nid];
  auto range_start = curr_nodes[nid].start;
  auto range_len = curr_nodes[nid].count;
  auto* col = input.data + split.colid * input.M;
  auto loffset = range_start, part = loffset + split.nLeft, roffset = part;
  auto end = range_start + range_len;
  int lflag = 0, rflag = 0, llen = 0, rlen = 0, minlen = 0;
  auto tid = threadIdx.x;
  while (loffset < part && roffset < end) {
    // find the samples in the left that belong to right and vice-versa
    auto loff = loffset + tid, roff = roffset + tid;
    if (llen == minlen)
      lflag = loff < part ? col[rowids[loff]] > split.quesval : 0;
    if (rlen == minlen)
      rflag = roff < end ? col[rowids[roff]] <= split.quesval : 0;
    // scan to compute the locations for each 'misfit' in the two partitions
    int lidx, ridx;
    BlockScanT(temp1).ExclusiveSum(lflag, lidx, llen);
    BlockScanT(temp2).ExclusiveSum(rflag, ridx, rlen);
    __syncthreads();
    minlen = llen < rlen ? llen : rlen;
    // compaction to figure out the right locations to swap
    if (lflag) lcomp[lidx] = loff;
    if (rflag) rcomp[ridx] = roff;
    __syncthreads();
    // reset the appropriate flags for the longer of the two
    if (lidx < minlen) lflag = 0;
    if (ridx < minlen) rflag = 0;
    if (llen == minlen) loffset += TPB;
    if (rlen == minlen) roffset += TPB;
    // swap the 'misfit's
    if (tid < minlen) {
      auto a = rowids[lcomp[tid]];
      auto b = rowids[rcomp[tid]];
      rowids[lcomp[tid]] = b;
      rowids[rcomp[tid]] = a;
    }
  }
  if (tid == 0) {
    curr_nodes[nid].makeChildNodes(n_nodes, total_nodes, next_nodes,
                                   splits[nid], n_depth);
  }
}

template <typename IdxT, typename LabelT, typename DataT, typename ObjectiveT,
          int TPB>
DI void computePrediction(IdxT range_start, IdxT range_len,
                          const Input<DataT, LabelT, IdxT>& input,
                          volatile Node<DataT, LabelT, IdxT>* nodes,
                          IdxT* n_leaves, void* smem) {
  using BinT = typename ObjectiveT::BinT;
  auto* shist = reinterpret_cast<BinT*>(smem);
  auto tid = threadIdx.x;
  for (int i = tid; i < input.numOutputs; i += blockDim.x) shist[i] = BinT();
  __syncthreads();
  auto len = range_start + range_len;
  for (auto i = range_start + tid; i < len; i += blockDim.x) {
    auto label = input.labels[input.rowids[i]];
    BinT::IncrementHistogram(shist, 1, 0, label);
  }
  __syncthreads();
  if (tid == 0) {
    auto pred = ObjectiveT::LeafPrediction(shist, input.numOutputs);
    nodes[0].makeLeaf(n_leaves, pred);
  }
}

template <typename DataT, typename LabelT, typename IdxT, typename ObjectiveT,
          int TPB>
__global__ void nodeSplitKernel(IdxT max_depth, IdxT min_samples_leaf,
                                IdxT min_samples_split, IdxT max_leaves,
                                DataT min_impurity_decrease,
                                Input<DataT, LabelT, IdxT> input,
                                volatile Node<DataT, LabelT, IdxT>* curr_nodes,
                                volatile Node<DataT, LabelT, IdxT>* next_nodes,
                                IdxT* n_nodes, const Split<DataT, IdxT>* splits,
                                IdxT* n_leaves, IdxT total_nodes,
                                IdxT* n_depth) {
  extern __shared__ char smem[];
  IdxT nid = blockIdx.x;
  volatile auto* node = curr_nodes + nid;
  auto range_start = node->start, n_samples = node->count;
  auto isLeaf = leafBasedOnParams<DataT, IdxT>(
    node->depth, max_depth, min_samples_split, max_leaves, n_leaves, n_samples);
  auto split = splits[nid];
  if (isLeaf || split.best_metric_val <= min_impurity_decrease ||
      split.nLeft < min_samples_leaf ||
      (n_samples - split.nLeft) < min_samples_leaf) {
    computePrediction<IdxT, LabelT, DataT, ObjectiveT, TPB>(
      range_start, n_samples, input, node, n_leaves, smem);
    return;
  }
  partitionSamples<DataT, LabelT, IdxT, TPB>(input, splits, curr_nodes,
                                             next_nodes, n_nodes, n_depth,
                                             total_nodes, (char*)smem);
}

/* Returns 'input' rounded up to a correctly-aligned pointer of type OutT* */
template <typename OutT, typename InT>
__device__ OutT* alignPointer(InT input) {
  return reinterpret_cast<OutT*>(
    raft::alignTo(reinterpret_cast<size_t>(input), sizeof(OutT)));
}

// 32-bit FNV1a hash
// Reference: http://www.isthe.com/chongo/tech/comp/fnv/index.html
const uint32_t fnv1a32_prime = uint32_t(16777619);
const uint32_t fnv1a32_basis = uint32_t(2166136261);

DI uint32_t fnv1a32(uint32_t hash, uint32_t txt) {
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

/**
 * @brief For a given values of (treeid, nodeid, seed), this function generates
 *        a unique permutation of [0, N - 1] values and returns 'k'th entry in
 *        from the permutation.
 * @return The 'k'th value from the permutation
 * @note This function does not allocated any temporary buffer, all the
 *       necessary values are recomputed.
 */
template <typename IdxT>
DI IdxT select(IdxT k, IdxT treeid, uint32_t nodeid, uint64_t seed, IdxT N) {
  __shared__ int blksum;
  uint32_t pivot_hash;
  int cnt = 0;

  if (threadIdx.x == 0) {
    blksum = 0;
  }
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

/**
 * @brief For every block, converts the smem pdf-histogram to
 *        cdf-histogram using inclusive block-sum-scan and returns
 *        the total_sum
 * @return The total sum aggregated over the sumscan,
 *         as well as the modified cdf-histogram pointer
 */
template <typename BinT, typename IdxT, int TPB>
DI BinT pdf_to_cdf(BinT* pdf_shist, BinT* cdf_shist, IdxT nbins) {
  // Blockscan instance preparation
  typedef cub::BlockScan<BinT, TPB> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // variable to accumulate aggregate of sumscans of previous iterations
  BinT total_aggregate = BinT();

  for (IdxT tix = threadIdx.x; tix < raft::ceildiv(nbins, TPB) * TPB;
       tix += blockDim.x) {
    BinT result;
    BinT block_aggregate;
    // getting the scanning element from pdf shist only
    BinT element = tix < nbins ? pdf_shist[tix] : BinT();
    // inclusive sum scan
    BlockScan(temp_storage).InclusiveSum(element, result, block_aggregate);
    __syncthreads();
    // store the result in cdf shist
    if (tix < nbins) {
      cdf_shist[tix] = result + total_aggregate;
    }
    total_aggregate += block_aggregate;
  }
  // return the total sum
  return total_aggregate;
}

template <typename DataT, typename LabelT, typename IdxT, int TPB,
          typename ObjectiveT, typename BinT>
__global__ void computeSplitKernel(
  BinT* hist, IdxT nbins, IdxT max_depth, IdxT min_samples_split,
  IdxT max_leaves, Input<DataT, LabelT, IdxT> input,
  const Node<DataT, LabelT, IdxT>* nodes, IdxT colStart, int* done_count,
  int* mutex, volatile Split<DataT, IdxT>* splits, ObjectiveT objective,
  IdxT treeid, WorkloadInfo<IdxT>* workload_info, uint64_t seed) {
  extern __shared__ char smem[];
  // Read workload info for this block
  WorkloadInfo<IdxT> workload_info_cta = workload_info[blockIdx.x];
  IdxT nid = workload_info_cta.nodeid;
  auto node = nodes[nid];
  auto range_start = node.start;
  auto range_len = node.count;

  IdxT offset_blockid = workload_info_cta.offset_blockid;
  IdxT num_blocks = workload_info_cta.num_blocks;

  auto end = range_start + range_len;
  auto pdf_shist_len = nbins * objective.NumClasses();
  auto cdf_shist_len = nbins * objective.NumClasses();
  auto* pdf_shist = alignPointer<BinT>(smem);
  auto* cdf_shist = alignPointer<BinT>(pdf_shist + pdf_shist_len);
  auto* sbins = alignPointer<DataT>(cdf_shist + cdf_shist_len);
  auto* sDone = alignPointer<int>(sbins + nbins);
  IdxT stride = blockDim.x * num_blocks;
  IdxT tid = threadIdx.x + offset_blockid * blockDim.x;

  // obtaining the feature to test split on
  IdxT col;
  if (input.nSampledCols == input.N) {
    col = colStart + blockIdx.y;
  } else {
    int colIndex = colStart + blockIdx.y;
    col = select(colIndex, treeid, node.info.unique_id, seed, input.N);
  }

  // populating shared memory with initial values
  for (IdxT i = threadIdx.x; i < pdf_shist_len; i += blockDim.x)
    pdf_shist[i] = BinT();
  for (IdxT j = threadIdx.x; j < cdf_shist_len; j += blockDim.x)
    cdf_shist[j] = BinT();
  for (IdxT b = threadIdx.x; b < nbins; b += blockDim.x)
    sbins[b] = input.quantiles[col * nbins + b];

  // synchronizing above changes across block
  __syncthreads();

  // compute pdf shared histogram for all bins for all classes in shared mem
  auto coloffset = col * input.M;
  for (auto i = range_start + tid; i < end; i += stride) {
    // each thread works over a data point and strides to the next
    auto row = input.rowids[i];
    auto d = input.data[row + coloffset];
    auto label = input.labels[row];
    IdxT bin =
      thrust::lower_bound(thrust::seq, sbins, sbins + nbins, d) - sbins;
    BinT::IncrementHistogram(pdf_shist, nbins, bin, label);
  }

  // synchronizeing above changes across block
  __syncthreads();
  if (num_blocks > 1) {
    // update the corresponding global location
    auto histOffset = ((nid * gridDim.y) + blockIdx.y) * pdf_shist_len;
    for (IdxT i = threadIdx.x; i < pdf_shist_len; i += blockDim.x) {
      BinT::AtomicAdd(hist + histOffset + i, pdf_shist[i]);
    }

    __threadfence();  // for commit guarantee
    __syncthreads();

    // last threadblock will go ahead and compute the best split
    bool last = true;
    last = MLCommon::signalDone(done_count + nid * gridDim.y + blockIdx.y,
                                num_blocks, offset_blockid == 0, sDone);
    // if not the last threadblock, exit
    if (!last) return;

    // store the complete global histogram in shared memory of last block
    for (IdxT i = threadIdx.x; i < pdf_shist_len; i += blockDim.x)
      pdf_shist[i] = hist[histOffset + i];

    __syncthreads();
  }

  /**
   * Scanning code:
   * span: block-wide
   * Function: convert the PDF calculated in the previous steps to CDF
   * This CDF is done over 2 passes
   * * one from left to right to sum-scan counts of left splits
   *   for each split-point.
   * * second from right to left to sum-scan the right splits
   *   for each split-point
   */
  for (IdxT c = 0; c < objective.NumClasses(); ++c) {
    /** left to right scan operation for scanning
     *  lesser-than-or-equal-to-bin counts **/
    // offsets to pdf and cdf shist pointers
    auto offset_pdf = nbins * c;
    auto offset_cdf = nbins * c;
    // converting pdf to cdf
    BinT total_sum = pdf_to_cdf<BinT, IdxT, TPB>(pdf_shist + offset_pdf,
                                                 cdf_shist + offset_cdf, nbins);
  }

  // create a split instance to test current feature split
  __syncthreads();

  // calculate the best candidate bins (one for each block-thread) in current feature and corresponding information gain for splitting
  Split<DataT, IdxT> sp =
    objective.Gain(cdf_shist, sbins, col, range_len, nbins);

  __syncthreads();

  // calculate best bins among candidate bins per feature using warp reduce
  // then atomically update across features to get best split per node
  // (in split[nid])
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

}  // namespace DecisionTree
}  // namespace ML
