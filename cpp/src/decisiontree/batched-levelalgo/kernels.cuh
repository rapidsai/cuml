/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <common/grid_sync.h>
#include <cuda_utils.h>
#include "input.cuh"
#include "node.cuh"
#include "split.cuh"

namespace ML {
namespace DecisionTree {

template <typename DataT, typename LabelT, typename IdxT>
__global__ void initialClassHistKernel(int* gclasshist, const IdxT* rowids,
                                       const LabelT* labels, IdxT nclasses,
                                       IdxT nrows) {
  extern __shared__ int* shist;
  for (IdxT i = threadIdx.x; i < nclasses; i += blockDim.x) shist = 0;
  __syncthreads();
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT stride = blockDim.x * gridDim.x;
  for (auto i = tid; i < nrows; i += stride) {
    auto row = rowids[i];
    auto label = labels[row];
    atomicAdd(shist + label, 1);
  }
  __syncthreads();
  for (IdxT i = threadIdx.x; i < nclasses; i += blockDim.x)
    atomicAdd(gclasshist + i, shist[i]);
}

/**
 * @brief Compute gain based on gini impurity metric
 * @param shist left/right class histograms for all bins (nbins x 2 x nclasses)
 *              After that, it also contains left/right sample counts for each
 *              of the bins (2 x nbins)
 * @param sbins quantiles for the current column (len = nbins)
 * @param parentGain parent node's best gain
 * @param sp will contain the per-thread best split so far
 * @param col current column
 * @param len total number of samples for the current node to be split
 * @param nbins number of bins
 * @param nclasses number of classes
 */
template <typename DataT, typename IdxT>
DI void giniInfoGain(const int* shist, const DataT* sbins, DataT parentGain,
                     Split<DataT, IdxT>& sp, IdxT col, IdxT len, IdxT nbins,
                     IdxT nclasses) {
  constexpr DataT One = DataT(1.0);
  DataT invlen = One / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    auto gain = DataT(0.0);
    auto nLeft = shist[nbins * 2 * nclasses + i];
    auto invLeft = One / nLeft;
    auto invRight = One / shist[nbins * 2 * nclasses + nbins + i];
    for (IdxT j = 0; j < nclasses; ++j) {
      auto lval = DataT(shist[i * 2 * nclasses + j]);
      gain += lval * invLeft * lval * invlen;
      auto rval = DataT(shist[i * 2 * nclasses + nclasses + j]);
      gain += rval * invRight * rval * invlen;
    }
    gain = parentGain - One + gain;
    sp.update(sbins[i], col, gain, nLeft);
  }
}

/**
 * @brief Decide whether the current node is to be declared as a leaf entirely
 *        based on the input hyper-params set by the user
 * @param myDepth depth of this node
 * @param max_depth maximum possible tree depth
 * @param min_rows_per_node min number of samples needed to split the node
 * @param max_leaves max leaf nodes per tree (it's a soft constraint)
 * @param n_leaves number of leaves in the tree already
 * @param nSamples number of samples belonging to this node
 * @return true if the current node is to be declared as a leaf, else false
 */
template <typename DataT, typename IdxT>
DI bool leafBasedOnParams(IdxT myDepth, IdxT max_depth, IdxT min_rows_per_node,
                          IdxT max_leaves, const IdxT* n_leaves,
                          IdxT nSamples) {
  if (myDepth < max_depth) return false;
  if (nSamples >= min_rows_per_node) return false;
  if (*n_leaves < max_leaves) return false;
  return true;
}

namespace internal {
struct Int2Max {
  int2 operator()(const int2& a, const int2& b) {
    int2 out;
    if (a.y > b.y)
      out = a;
    else if (a.y == b.y && a.x < b.x)
      out = a;
    else
      out = b;
    return out;
  }
};
};  // namespace internal

/**
 * @brief Compute the prediction value for the current leaf node
 * @note to be called by only one block from all participating blocks
 *       'smem' must be atleast of size `sizeof(int) * input.nclasses`
 */
///@todo: support for regression
template <typename DataT, typename LabelT, typename IdxT, int TPB>
DI void computePrediction(IdxT range_start, IdxT range_len,
                          const Input<DataT, LabelT, IdxT>& input,
                          volatile Node<DataT, LabelT, IdxT>* nodes,
                          volatile IdxT* n_leaves, void* smem) {
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
  auto op = internal::Int2Max();
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
                         Node<DataT, LabelT, IdxT>* curr_nodes,
                         Node<DataT, LabelT, IdxT>* next_nodes,
                         volatile IdxT* n_nodes, IdxT total_nodes, char* smem) {
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
  auto range_len = curr_nodes[nid].end;
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
    if (lflag) lcomp[lidx] = lidx + loffset;
    if (rflag) rcomp[ridx] = ridx + roffset;
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
                                   splits[nid]);
  }
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
__global__ void nodeSplitKernel(IdxT max_depth, IdxT min_rows_per_node,
                                IdxT max_leaves, DataT min_impurity_decrease,
                                Input<DataT, LabelT, IdxT> input,
                                volatile Node<DataT, LabelT, IdxT>* curr_nodes,
                                volatile Node<DataT, LabelT, IdxT>* next_nodes,
                                volatile IdxT* n_nodes,
                                const Split<DataT, IdxT>* splits,
                                volatile IdxT* n_leaves, IdxT total_nodes) {
  extern __shared__ char smem[];
  IdxT nid = blockIdx.x;
  auto node = curr_nodes[nid];
  auto range_start = node.start, range_len = node.end;
  auto isLeaf = leafBasedOnParams<DataT, IdxT>(
    node.depth, max_depth, min_rows_per_node, max_leaves, n_leaves, range_len);
  if (isLeaf || splits[nid].gain < min_impurity_decrease) {
    computePrediction<DataT, LabelT, IdxT>(range_start, range_len, input,
                                           curr_nodes + nid, n_leaves, smem);
    return;
  }
  partitionSamples<DataT, LabelT, IdxT, TPB>(
    input, splits, curr_nodes, next_nodes, n_nodes, total_nodes, smem);
}

///@todo: support regression
///@todo: support other metrics
///@todo: special-case this for gridDim.x == 1
template <typename DataT, typename LabelT, typename IdxT, int TPB>
__global__ void computeSplitKernel(int* hist, IdxT nbins, IdxT max_depth,
                                   IdxT min_rows_per_node, IdxT max_leaves,
                                   Input<DataT, LabelT, IdxT> input,
                                   const Node<DataT, LabelT, IdxT>* nodes,
                                   IdxT colStart, int* done_count, int* mutex,
                                   const IdxT* n_leaves,
                                   Split<DataT, IdxT>* splits) {
  extern __shared__ char smem[];
  IdxT nid = blockIdx.z;
  auto node = nodes[nid];
  auto range_start = node.start;
  auto range_len = node.end;
  if (leafBasedOnParams<DataT, IdxT>(node.depth, max_depth, min_rows_per_node,
                                     max_leaves, n_leaves, range_len)) {
    return;
  }
  auto parentGain = node.parentGain;
  auto end = range_start + range_len;
  auto nclasses = input.nclasses;
  auto binSize = nbins * 2 * nclasses;
  auto len = binSize + 2 * nbins;
  auto* shist = reinterpret_cast<int*>(smem);
  auto* sbins = reinterpret_cast<DataT*>(smem + sizeof(int) * len);
  IdxT stride = blockDim.x * gridDim.x;
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto col = input.colids[colStart + blockIdx.y];
  if (col >= input.nSampledCols) return;
  for (IdxT i = 0; i < len; i += blockDim.x) shist[i] = 0;
  for (IdxT b = 0; b < nbins; b += blockDim.x)
    sbins[b] = input.quantiles[col * nbins + b];
  __syncthreads();
  auto coloffset = col * input.M;
  // compute class histogram for all bins for all classes in shared mem
  for (auto i = range_start + tid; i < end; i += stride) {
    auto row = input.rowids[i];
    auto d = input.data[row + coloffset];
    auto label = input.labels[row];
    for (IdxT b = 0; b < nbins; ++b) {
      auto isLeft = d <= sbins[b];  // no divergence
      auto offset = b * 2 * nclasses + isLeft * nclasses + label;
      atomicAdd(shist + offset, 1);  // class hist
      offset = binSize + isLeft * nbins + b;
      atomicAdd(shist + offset, 1);  // sample count
    }
  }
  __syncthreads();
  // update the corresponding global location
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    atomicAdd(hist + nid * len + i, shist[i]);
  }
  __syncthreads();
  // last threadblock will go ahead and compute the best split
  auto last = MLCommon::signalDone(done_count + nid * gridDim.y + blockIdx.y,
                                   gridDim.x, blockIdx.x == 0, smem);
  if (!last) return;
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    shist[i] = hist[nid * len + i];
  }
  __syncthreads();
  Split<DataT, IdxT> sp;
  sp.init();
  giniInfoGain<DataT, LabelT, IdxT>(shist, sbins, parentGain, sp, col,
                                    range_len, nbins, nclasses);
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

}  // namespace DecisionTree
}  // namespace ML
