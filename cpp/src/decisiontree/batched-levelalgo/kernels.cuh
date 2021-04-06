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
 * @brief Traits used to customize device-side methods for classification task
 *
 * @tparam _data  data type
 * @tparam _label label type
 * @tparam _idx   index type
 * @tparam TPB    threads per block
 */
template <typename _data, typename _label, typename _idx, int TPB>
struct ClsDeviceTraits {
  typedef _data DataT;
  typedef _label LabelT;
  typedef _idx IdxT;

 private:
  struct Int2Max {
    DI int2 operator()(const int2& a, const int2& b) {
      int2 out;
      if (a.y > b.y)
        out = a;
      else if (a.y == b.y && a.x < b.x)
        out = a;
      else
        out = b;
      return out;
    }
  };  // struct Int2Max

 public:
  /**
   * @note to be called by only one block from all participating blocks
   *       'smem' must be atleast of size `sizeof(int) * input.nclasses`
   */
  static DI void computePrediction(IdxT range_start, IdxT range_len,
                                   const Input<DataT, LabelT, IdxT>& input,
                                   volatile Node<DataT, LabelT, IdxT>* nodes,
                                   IdxT* n_leaves, void* smem) {
    typedef cub::BlockReduce<int2, TPB> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp;
    auto* shist = reinterpret_cast<int*>(smem);
    auto tid = threadIdx.x;
    for (int i = tid; i < input.nclasses; i += blockDim.x) shist[i] = 0;
    __syncthreads();
    auto len = range_start + range_len;
    for (auto i = range_start + tid; i < len; i += blockDim.x) {
      auto label = input.labels[input.rowids[i]];
      atomicAdd(shist + label, 1);
    }
    __syncthreads();
    auto op = Int2Max();
    int2 v = {-1, -1};
    for (int i = tid; i < input.nclasses; i += blockDim.x) {
      int2 tmp = {i, shist[i]};
      v = op(v, tmp);
    }
    v = BlockReduceT(temp).Reduce(v, op);
    __syncthreads();
    if (tid == 0) {
      nodes[0].makeLeaf(n_leaves, LabelT(v.x));
    }
  }
};  // struct ClsDeviceTraits

/**
 * @brief Traits used to customize device-side methods for regression task
 *
 * @tparam _data  data type
 * @tparam _label label type
 * @tparam _idx   index type
 * @tparam TPB    threads per block
 */
template <typename _data, typename _label, typename _idx, int TPB>
struct RegDeviceTraits {
  typedef _data DataT;
  typedef _label LabelT;
  typedef _idx IdxT;

  /**
   * @note to be called by only one block from all participating blocks
   *       'smem' is not used, but kept for the sake of interface parity with
   *       the corresponding method for classification
   */
  static DI void computePrediction(IdxT range_start, IdxT range_len,
                                   const Input<DataT, LabelT, IdxT>& input,
                                   volatile Node<DataT, LabelT, IdxT>* nodes,
                                   IdxT* n_leaves, void* smem) {
    typedef cub::BlockReduce<LabelT, TPB> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp;
    LabelT sum = LabelT(0.0);
    auto tid = threadIdx.x;
    auto len = range_start + range_len;
    for (auto i = range_start + tid; i < len; i += blockDim.x) {
      auto label = input.labels[input.rowids[i]];
      sum += label;
    }
    sum = BlockReduceT(temp).Sum(sum);
    __syncthreads();
    if (tid == 0) {
      if (range_len != 0) {
        nodes[0].makeLeaf(n_leaves, sum / range_len);
      } else {
        nodes[0].makeLeaf(n_leaves, 0.0);
      }
    }
  }
};  // struct RegDeviceTraits

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
DI bool leafBasedOnParams(IdxT myDepth, IdxT max_depth, IdxT min_samples_split,
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

template <typename DataT, typename LabelT, typename IdxT, typename DevTraits,
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
    DevTraits::computePrediction(range_start, n_samples, input, node, n_leaves,
                                 smem);
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
template <typename DataT, typename IdxT, int TPB>
DI DataT pdf_to_cdf(DataT* pdf_shist, DataT* cdf_shist, IdxT nbins) {
  // Blockscan instance preparation
  typedef cub::BlockScan<DataT, TPB> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // variable to accumulate aggregate of sumscans of previous iterations
  DataT total_aggregate = DataT(0);

  for (IdxT tix = threadIdx.x; tix < max(TPB, nbins); tix += blockDim.x) {
    DataT result;
    DataT block_aggregate;
    // getting the scanning element from pdf shist only
    DataT element = tix < nbins ? pdf_shist[tix] : 0;
    // inclusive sum scan
    BlockScan(temp_storage).InclusiveSum(element, result, block_aggregate);
    __syncthreads();
    // store the result in cdf shist
    if (tix < nbins) {
      cdf_shist[tix] = result + total_aggregate;
      total_aggregate += block_aggregate;
    }
  }
  // return the total sum
  return total_aggregate;
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
__global__ void computeSplitClassificationKernel(
  int* hist, IdxT nbins, IdxT max_depth, IdxT min_samples_split,
  IdxT min_samples_leaf, DataT min_impurity_decrease, IdxT max_leaves,
  Input<DataT, LabelT, IdxT> input, const Node<DataT, LabelT, IdxT>* nodes,
  IdxT colStart, int* done_count, int* mutex, const IdxT* n_leaves,
  Split<DataT, IdxT>* splits, CRITERION splitType, IdxT treeid, uint64_t seed) {
  extern __shared__ char smem[];
  IdxT nid = blockIdx.z;
  auto node = nodes[nid];
  auto range_start = node.start;
  auto range_len = node.count;

  // return if leaf
  if (leafBasedOnParams<DataT, IdxT>(node.depth, max_depth, min_samples_split,
                                     max_leaves, n_leaves, range_len)) {
    return;
  }
  auto end = range_start + range_len;
  auto nclasses = input.nclasses;
  auto pdf_shist_len = (nbins + 1) * nclasses;
  auto cdf_shist_len = nbins * 2 * nclasses;
  auto* pdf_shist = alignPointer<int>(smem);
  auto* cdf_shist = alignPointer<int>(pdf_shist + pdf_shist_len);
  auto* sbins = alignPointer<DataT>(cdf_shist + cdf_shist_len);
  auto* sDone = alignPointer<int>(sbins + nbins);
  IdxT stride = blockDim.x * gridDim.x;
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;

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
    pdf_shist[i] = 0;
  for (IdxT j = threadIdx.x; j < cdf_shist_len; j += blockDim.x)
    cdf_shist[j] = 0;
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
    for (IdxT b = 0; b < nbins; ++b) {
      if (d <= sbins[b]) {
        auto offset = label * (1 + nbins) + b;
        atomicAdd(pdf_shist + offset, 1);
        break;
      }
    }
  }

  // synchronizeing above changes across block
  __syncthreads();

  // update the corresponding global location
  auto histOffset = ((nid * gridDim.y) + blockIdx.y) * pdf_shist_len;
  for (IdxT i = threadIdx.x; i < pdf_shist_len; i += blockDim.x) {
    atomicAdd(hist + histOffset + i, pdf_shist[i]);
  }

  __threadfence();  // for commit guarantee
  __syncthreads();

  // last threadblock will go ahead and compute the best split
  bool last = true;
  if (gridDim.x > 1) {
    last = MLCommon::signalDone(done_count + nid * gridDim.y + blockIdx.y,
                                gridDim.x, blockIdx.x == 0, sDone);
  }
  // if not the last threadblock, exit
  if (!last) return;

  // store the complete global histogram in shared memory of last block
  for (IdxT i = threadIdx.x; i < pdf_shist_len; i += blockDim.x)
    pdf_shist[i] = hist[histOffset + i];

  __syncthreads();

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
  for (IdxT c = 0; c < nclasses; ++c) {
    /** left to right scan operation for scanning
     *  lesser-than-or-equal-to-bin counts **/
    // offsets to pdf and cdf shist pointers
    auto offset_pdf = (1 + nbins) * c;
    auto offset_cdf = (2 * nbins) * c;
    // converting pdf to cdf
    int total_sum = pdf_to_cdf<int, IdxT, TPB>(pdf_shist + offset_pdf,
                                               cdf_shist + offset_cdf, nbins);

    // greater-than split starts after nbins of less-than-equal split
    // locations
    offset_cdf += nbins;
    /** samples that are greater-than-bin calculated by difference
     *  of count of lesser-than-equal samples from total_sum.
     **/
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      *(cdf_shist + offset_cdf + i) =
        total_sum - *(cdf_shist + 2 * nbins * c + i);
    }
  }

  // create a split instance to test current feature split
  Split<DataT, IdxT> sp;
  sp.init();
  __syncthreads();

  // calculate the best candidate bins (one for each block-thread) in current feature and corresponding information gain for splitting
  if (splitType == CRITERION::GINI) {
    giniGain<DataT, IdxT>(cdf_shist, sbins, sp, col, range_len, nbins, nclasses,
                          min_samples_leaf, min_impurity_decrease);
  } else {
    entropyGain<DataT, IdxT>(cdf_shist, sbins, sp, col, range_len, nbins,
                             nclasses, min_samples_leaf, min_impurity_decrease);
  }
  __syncthreads();

  // calculate best bins among candidate bins per feature using warp reduce
  // then atomically update across features to get best split per node (in split[nid])
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
__global__ void computeSplitRegressionKernel(
  DataT* pred, DataT* pred2, DataT* pred2P, IdxT* count, IdxT nbins,
  IdxT max_depth, IdxT min_samples_split, IdxT min_samples_leaf,
  DataT min_impurity_decrease, IdxT max_leaves,
  Input<DataT, LabelT, IdxT> input, const Node<DataT, LabelT, IdxT>* nodes,
  IdxT colStart, int* done_count, int* mutex, const IdxT* n_leaves,
  Split<DataT, IdxT>* splits, void* workspace, CRITERION splitType, IdxT treeid,
  uint64_t seed) {
  extern __shared__ char smem[];
  IdxT nid = blockIdx.z;
  auto node = nodes[nid];
  auto range_start = node.start;
  auto range_len = node.count;

  // exit if current node is leaf
  if (leafBasedOnParams<DataT, IdxT>(node.depth, max_depth, min_samples_split,
                                     max_leaves, n_leaves, range_len)) {
    return;
  }

  // variables
  auto end = range_start + range_len;
  auto len = nbins * 2;
  auto pdf_spred_len = 1 + nbins;
  auto cdf_spred_len = 2 * nbins;
  IdxT stride = blockDim.x * gridDim.x;
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT col;

  // allocating pointers to shared memory
  auto* pdf_spred = alignPointer<DataT>(smem);
  auto* cdf_spred = alignPointer<DataT>(pdf_spred + pdf_spred_len);
  auto* pdf_scount = alignPointer<int>(cdf_spred + cdf_spred_len);
  auto* cdf_scount = alignPointer<int>(pdf_scount + nbins);
  auto* sbins = alignPointer<DataT>(cdf_scount + nbins);
  auto* spred2 = alignPointer<DataT>(sbins + nbins);
  auto* spred2P = alignPointer<DataT>(spred2 + len);
  auto* spredP = alignPointer<DataT>(spred2P + nbins);
  auto* sDone = alignPointer<int>(spredP + nbins);

  // select random feature to split-check
  // (if feature-sampling is true)
  if (input.nSampledCols == input.N) {
    col = colStart + blockIdx.y;
  } else {
    int colIndex = colStart + blockIdx.y;
    col = select(colIndex, treeid, node.info.unique_id, seed, input.N);
  }

  // memset smem pointers
  for (IdxT i = threadIdx.x; i < pdf_spred_len; i += blockDim.x) {
    pdf_spred[i] = DataT(0.0);
  }
  for (IdxT i = threadIdx.x; i < cdf_spred_len; i += blockDim.x) {
    cdf_spred[i] = DataT(0.0);
  }
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    pdf_scount[i] = 0;
    cdf_scount[i] = 0;
    sbins[i] = input.quantiles[col * nbins + i];
  }
  __syncthreads();
  auto coloffset = col * input.M;

  // compute prediction pdfs and count pdfs
  for (auto i = range_start + tid; i < end; i += stride) {
    auto row = input.rowids[i];
    auto d = input.data[row + coloffset];
    auto label = input.labels[row];
    for (IdxT b = 0; b < nbins; ++b) {
      // if sample is less-than-or-equal to threshold
      if (d <= sbins[b]) {
        atomicAdd(pdf_spred + b, label);
        atomicAdd(pdf_scount + b, 1);
        break;
      }
    }
  }
  __syncthreads();

  // update the corresponding global location for counts
  auto gcOffset = ((nid * gridDim.y) + blockIdx.y) * nbins;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    atomicAdd(count + gcOffset + i, pdf_scount[i]);
  }

  // update the corresponding global location for preds
  auto gOffset = ((nid * gridDim.y) + blockIdx.y) * pdf_spred_len;
  for (IdxT i = threadIdx.x; i < pdf_spred_len; i += blockDim.x) {
    atomicAdd(pred + gOffset + i, pdf_spred[i]);
  }
  __threadfence();  // for commit guarantee
  __syncthreads();

  // Wait until all blockIdx.x's are done
  MLCommon::GridSync gs(workspace, MLCommon::SyncType::ACROSS_X, false);
  gs.sync();

  // transfer from global to smem
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    pdf_scount[i] = count[gcOffset + i];
    spred2P[i] = DataT(0.0);
  }
  for (IdxT i = threadIdx.x; i < pdf_spred_len; i += blockDim.x) {
    pdf_spred[i] = pred[gOffset + i];
  }
  // memset spred2
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    spred2[i] = DataT(0.0);
  }
  __syncthreads();

  /** pdf to cdf conversion **/

  /** get cdf of spred from pdf_spred **/
  // cdf of samples lesser-than-equal to threshold
  DataT total_sum = pdf_to_cdf<DataT, IdxT, TPB>(pdf_spred, cdf_spred, nbins);

  // cdf of samples greater than threshold
  // calculated by subtracting lesser-than-equals from total_sum
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    *(cdf_spred + nbins + i) = total_sum - *(cdf_spred + i);
  }

  /** get cdf of scount from pdf_scount **/
  pdf_to_cdf<int, IdxT, TPB>(pdf_scount, cdf_scount, nbins);
  __syncthreads();

  // calcualting prediction average-sums
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    spredP[i] = cdf_spred[i] + cdf_spred[i + nbins];
  }
  __syncthreads();

  // now, compute the mean value to be used for metric update
  auto invlen = DataT(1.0) / range_len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    auto cnt_l = DataT(cdf_scount[i]);
    auto cnt_r = DataT(range_len - cdf_scount[i]);
    cdf_spred[i] /= cnt_l;
    cdf_spred[i + nbins] /= cnt_r;
    spredP[i] *= invlen;
  }
  __syncthreads();

  /* Make a second pass over the data to compute gain */

  // 2nd pass over data to compute partial metric across blockIdx.x's
  if (splitType == CRITERION::MAE) {
    for (auto i = range_start + tid; i < end; i += stride) {
      auto row = input.rowids[i];
      auto d = input.data[row + coloffset];
      auto label = input.labels[row];
      for (IdxT b = 0; b < nbins; ++b) {
        auto isRight = d > sbins[b];  // no divergence
        auto offset = isRight * nbins + b;
        auto diff = label - (isRight ? cdf_spred[nbins + b] : cdf_spred[b]);
        atomicAdd(spred2 + offset, raft::myAbs(diff));
        atomicAdd(spred2P + b, raft::myAbs(label - spredP[b]));
      }
    }
  } else {
    for (auto i = range_start + tid; i < end; i += stride) {
      auto row = input.rowids[i];
      auto d = input.data[row + coloffset];
      auto label = input.labels[row];
      for (IdxT b = 0; b < nbins; ++b) {
        auto isRight = d > sbins[b];  // no divergence
        auto offset = isRight * nbins + b;
        auto diff = label - (isRight ? cdf_spred[nbins + b] : cdf_spred[b]);
        auto diff2 = label - spredP[b];
        atomicAdd(spred2 + offset, (diff * diff));
        atomicAdd(spred2P + b, (diff2 * diff2));
      }
    }
  }
  __syncthreads();

  // update the corresponding global location for pred2P
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    atomicAdd(pred2P + gcOffset + i, spred2P[i]);
  }

  // changing gOffset for pred2 from that of pred
  gOffset = ((nid * gridDim.y) + blockIdx.y) * len;
  // update the corresponding global location for pred2
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    atomicAdd(pred2 + gOffset + i, spred2[i]);
  }
  __threadfence();  // for commit guarantee
  __syncthreads();

  // last threadblock will go ahead and compute the best split
  bool last = true;
  if (gridDim.x > 1) {
    last = MLCommon::signalDone(done_count + nid * gridDim.y + blockIdx.y,
                                gridDim.x, blockIdx.x == 0, sDone);
  }

  // exit if not last
  if (!last) return;

  // last block computes the final gain
  // create a split instance to test current feature split
  Split<DataT, IdxT> sp;
  sp.init();

  // store global pred2 and pred2P into shared memory of last x-dim block
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    spred2[i] = pred2[gOffset + i];
  }
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    spred2P[i] = pred2P[gcOffset + i];
  }
  __syncthreads();

  // calculate the best candidate bins (one for each block-thread) in current
  // feature and corresponding regression-metric gain for splitting
  regressionMetricGain(spred2, spred2P, cdf_scount, sbins, sp, col, range_len,
                       nbins, min_samples_leaf, min_impurity_decrease);
  __syncthreads();

  // calculate best bins among candidate bins per feature using warp reduce
  // then atomically update across features to get best split per node (in split[nid])
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

}  // namespace DecisionTree
}  // namespace ML
