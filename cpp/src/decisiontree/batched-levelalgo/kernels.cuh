/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

DI uint32_t mulwide32(uint32_t a, uint32_t b, uint32_t *hi) { 
    uint64_t res = uint64_t(a) * b;
    *hi = uint32_t(res >> 32);
    return uint32_t(res & 0xFFFFFFFF);
}

// Philox 4x32 constants
constexpr uint32_t ADD_CONSTANT_0 = uint32_t(0x9E3779B9);
constexpr uint32_t ADD_CONSTANT_1 = uint32_t(0xBB67AE85);

constexpr uint32_t MUL_CONSTANT_0 = uint32_t(0xD2511F53);
constexpr uint32_t MUL_CONSTANT_1 = uint32_t(0xCD9E8D57);


DI uint4 philox_4x32_round(uint4 counter, uint2 seed) { 
    uint32_t hi0, hi1;
    uint4 out;
    uint32_t lo0 = mulwide32(MUL_CONSTANT_0, counter.x, &hi0);
    uint32_t lo1 = mulwide32(MUL_CONSTANT_1, counter.z, &hi1);
    out.x = hi1 ^ counter.y ^ seed.x;
    out.y = lo1;
    out.z = hi0 ^ counter.w ^ seed.y;
    out.w = lo0;
    return out;
}

DI uint4 philox_4x32_10(uint4 counter, uint2 seed) { 
    counter = philox_4x32_round(counter, seed); // 0
    seed.x += ADD_CONSTANT_0;
    seed.y += ADD_CONSTANT_1;
    counter = philox_4x32_round(counter, seed); // 1
    seed.x += ADD_CONSTANT_0;
    seed.y += ADD_CONSTANT_1;
    counter = philox_4x32_round(counter, seed); // 2
    seed.x += ADD_CONSTANT_0;
    seed.y += ADD_CONSTANT_1;
    counter = philox_4x32_round(counter, seed); // 3
    seed.x += ADD_CONSTANT_0;
    seed.y += ADD_CONSTANT_1;
    counter = philox_4x32_round(counter, seed); // 4
    seed.x += ADD_CONSTANT_0;
    seed.y += ADD_CONSTANT_1;
    counter = philox_4x32_round(counter, seed); // 5
    seed.x += ADD_CONSTANT_0;
    seed.y += ADD_CONSTANT_1;
    counter = philox_4x32_round(counter, seed); // 6
    seed.x += ADD_CONSTANT_0;
    seed.y += ADD_CONSTANT_1;
    counter = philox_4x32_round(counter, seed); // 7
    seed.x += ADD_CONSTANT_0;
    seed.y += ADD_CONSTANT_1;
    counter = philox_4x32_round(counter, seed); // 8
    seed.x += ADD_CONSTANT_0;
    seed.y += ADD_CONSTANT_1;
    counter = philox_4x32_round(counter, seed); // 9

    return counter;
}

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

template <typename IdxT>
DI IdxT selectFeature(IdxT treeid, IdxT nodeid, IdxT k, uint64_t seed, IdxT N) {
  uint4 counter;
  uint2 key;

  counter.x = uint32_t(k);
  counter.y = uint32_t(nodeid);
  counter.z = uint32_t(treeid);
  counter.w = uint32_t(0xAAAAAAAA); // Reserved capacity
  key.x = uint32_t(seed & 0xFFFFFFFF);
  key.y = uint32_t(seed >> 32);

  counter.x = k;
  uint4 n_counter = philox_4x32_10(counter, key);
  uint32_t watch_for = k + n_counter.x % (N - k);

  for(int i = k - 1; i >= 0; i--) {
    counter.x = i;
    n_counter = philox_4x32_10(counter, key);
    int  j = i + n_counter.x % (N - i);
    if(j == watch_for) {
        watch_for = i;
    }
  }
  return IdxT(watch_for);
}

DI uint32_t kiss99(uint32_t z, uint32_t w, uint32_t jsr, uint32_t jcong) {
    uint32_t MWC;
    z = 36969*(z&65535) + (z >> 16);
    w = 18000*(w&65535) + (w >> 16);
    MWC = ((z << 16) + w);
    jsr ^= (jsr << 17);
    jsr ^= (jsr >> 13);
    jsr ^= (jsr << 5);
    jcong = 69069*jcong + 1234567;
    return ((MWC ^ jcong) + jsr);
}

template <typename IdxT>
DI IdxT selectFeature2(IdxT treeid, IdxT nodeid, IdxT k, uint64_t seed, IdxT N) {

  // typedef cub::BlockScan<int, TPB> BlockScanT;
  // __shared__ typename BlockScanT::TempStorage temp;

  __shared__ int total_count;
  int trial_count;
  uint32_t toss;
  int cnt;
  trial_count = 0;
  while(1) {
    if(threadIdx.x == 0) {
      total_count = 0;
    }
    __syncthreads();

    cnt = 0;
    for(int i = threadIdx.x; i < N; i += blockDim.x) {
      toss = kiss99(uint32_t(treeid), uint32_t(nodeid), uint32_t(seed >> 32) ^ trial_count, 
                    uint32_t(seed) ^ i);
      toss = toss % N;
      if(toss < N/2) {
        cnt++;
      }
    }
    atomicAdd(&total_count, cnt);
    __syncthreads();
    if(total_count >= N/2) {
      break;
    }
    trial_count++;
  }

  return ;
}

const uint32_t fnv1a32_prime = uint32_t(16777619);
const uint32_t fnv1a32_basis = uint32_t(2166136261);

DI uint32_t fnv1a32(uint32_t hash, uint32_t txt)
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

template <typename IdxT>
DI IdxT selectFeature3(IdxT treeid, IdxT nodeid, IdxT k, uint64_t seed, IdxT N) {
   __shared__ int blksum;
  uint32_t pivote;
  int cnt = 0;

  if (threadIdx.x == 0) {
    blksum = 0;
  }
  pivote = fnv1a32_basis;
  pivote = fnv1a32(pivote, uint32_t(k));
  pivote = fnv1a32(pivote, uint32_t(treeid));
  pivote = fnv1a32(pivote, uint32_t(nodeid));
  pivote = fnv1a32(pivote, uint32_t(seed >> 32));
  pivote = fnv1a32(pivote, uint32_t(seed));

  uint32_t j;
  for(int i = threadIdx.x; i < N; i += blockDim.x) {
    if(i == k) continue;
    j = fnv1a32_basis;
    j = fnv1a32(j, uint32_t(i));
    j = fnv1a32(j, uint32_t(treeid));
    j = fnv1a32(j, uint32_t(nodeid));
    j = fnv1a32(j, uint32_t(seed >> 32));
    j = fnv1a32(j, uint32_t(seed));

    if(j < pivote) 
      cnt++;
    else if (j == pivote && i < k)
      cnt++;
  }
  __syncthreads();
  if(cnt > 0)
    atomicAdd(&blksum, cnt);
  __syncthreads();
  return blksum;
}


template <typename DataT, typename LabelT, typename IdxT, int TPB>
__global__ void computeSplitClassificationKernel(
  int* hist, IdxT nbins, IdxT max_depth, IdxT min_samples_split,
  IdxT max_leaves, Input<DataT, LabelT, IdxT> input,
  const Node<DataT, LabelT, IdxT>* nodes, IdxT colStart, int* done_count,
  int* mutex, const IdxT* n_leaves, Split<DataT, IdxT>* splits,
  CRITERION splitType, IdxT treeid, uint64_t seed) {
  extern __shared__ char smem[];
  IdxT nid = blockIdx.z;
  auto node = nodes[nid];
  auto range_start = node.start;
  auto range_len = node.count;
  if (leafBasedOnParams<DataT, IdxT>(node.depth, max_depth, min_samples_split,
                                     max_leaves, n_leaves, range_len)) {
    return;
  }
  auto end = range_start + range_len;
  auto nclasses = input.nclasses;
  auto len = nbins * 2 * nclasses;
  auto* shist = alignPointer<int>(smem);
  auto* sbins = alignPointer<DataT>(shist + len);
  auto* sDone = alignPointer<int>(sbins + nbins);
  IdxT stride = blockDim.x * gridDim.x;
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Approach 0
  // auto col = input.colids[colStart + blockIdx.y];
 
  // Approach 1
  int colIndex = colStart + blockIdx.y;
  auto col = selectFeature(treeid, int(node.info.unique_id), colIndex, seed, input.N);

  // Approach 2
  // int colIndex = colStart + blockIdx.y;
  // auto col = selectFeature2(treeid, int(node.info.unique_id), colIndex, seed, input.N);

  // Approach 3
  // int colIndex = colStart + blockIdx.y;
  // auto col = selectFeature3(treeid, int(node.info.unique_id), colIndex, seed, input.N);


  // if(tid == 0 && node.info.unique_id == NODE_TO_PRINT) {
  //   // printf("%d <-> %d, ", blockIdx.y, col);
  //   printf("%d, ", col);
  // }

  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) shist[i] = 0;
  for (IdxT b = threadIdx.x; b < nbins; b += blockDim.x)
    sbins[b] = input.quantiles[col * nbins + b];
  __syncthreads();
// #if defined(RF_PRINT_DEVICE_DEBUG_MSGS)
  // if(node.info.unique_id == NODE_TO_PRINT /*&& col == COL_TO_PRINT*/) {
    if(tid == 0) {
      // printf("treeid = %.3d, unique_id = %.3d, colIndex = %.3d, seed = %.10lu,"
      //        "N = %.3d, col = %.3d\n", treeid, int(node.info.unique_id), colIndex,
      //         seed, input.N, col);

      // printf("treeid = %d, nodeid = %d, k = %d, seed = %lu, N = %d watch_for = %d\n",
        // treeid, nid, colIndex, seed, input.N, col);
      // printf("At blockIdx.x = %d, evaluating split for col = %d\n"
      //        "Considering samples rowids[%d] to rowids[%d]\n",
      //        blockIdx.x, col, range_start, end);
      // for(int i = range_start; i < end; i++) {
      //   printf("[[%d, %d, %f, %d]], ", i, input.rowids[i], 
      //     input.data[input.rowids[i] + col * input.M],
      //     input.labels[input.rowids[i]]);
      //   }
      // printf("Questions considered for the split:\n");
      // for(int i = 0; i < nbins; i++)
      //   printf("quesval[%d]: %f\n", i, sbins[i]);
    }
  // }
// #endif
  auto coloffset = col * input.M;
  // compute class histogram for all bins for all classes in shared mem
  for (auto i = range_start + tid; i < end; i += stride) {
    auto row = input.rowids[i];
    auto d = input.data[row + coloffset];
    auto label = input.labels[row];
    for (IdxT b = 0; b < nbins; ++b) {
      auto isRight = d > sbins[b];  // no divergence
      auto offset = b * 2 * nclasses + isRight * nclasses + label;
      atomicAdd(shist + offset, 1);  // class hist
    }
  }
  __syncthreads();
#if defined(RF_PRINT_DEVICE_DEBUG_MSGS)
  // if(node.info.unique_id == NODE_TO_PRINT && col == COL_TO_PRINT) {
  //     if(tid == 0) {
  //       for(int i = 0; i < len; i++)
  //         printf("shist[%d] = %d\n", i, shist[i]);
  //     }
  // }
#endif
  // update the corresponding global location
  auto histOffset = ((nid * gridDim.y) + blockIdx.y) * len;
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    atomicAdd(hist + histOffset + i, shist[i]);
  }
  __threadfence();  // for commit guarantee
  __syncthreads();
#if defined(RF_PRINT_DEVICE_DEBUG_MSGS)
  if(node.info.unique_id == NODE_TO_PRINT && col == COL_TO_PRINT) {
    if(tid == 0) {
      for(int bin_id = 0; bin_id < nbins; bin_id++) {
        printf("quesval[%d]: %f\n", bin_id, sbins[bin_id]);
        for(int cls = 0; cls < nclasses; cls++) {
          int ncols= gridDim.y;
          int left_count  = hist[cls + 0*nclasses + bin_id*2*nclasses +
                            col*nbins*2*nclasses + 
                            nid*ncols*nbins*2*nclasses];

          int right_count = hist[cls + 1*nclasses + bin_id*2*nclasses +
                            col*nbins*2*nclasses +
                            nid*ncols*nbins*2*nclasses];
          printf("\tFor class = %d, left = %d, right = %d\n", cls, 
                 left_count, right_count);
        }
      }
    }
  }
#endif
  __syncthreads();
  // last threadblock will go ahead and compute the best split
  bool last = true;
  if (gridDim.x > 1) {
    last = MLCommon::signalDone(done_count + nid * gridDim.y + blockIdx.y,
                                gridDim.x, blockIdx.x == 0, sDone);
  }
  if (!last) return;
#if defined(RF_PRINT_DEVICE_DEBUG_MSGS)
  // if(threadIdx.x == 0)
  //   printf("threadIdx{.x, .y, .z} = {%d, %d, %d}, blockdIdx{.x, .y, .z} = {%d, %d, %d}\n", 
  //     threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
#endif
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x)
    shist[i] = hist[histOffset + i];
  Split<DataT, IdxT> sp;
  sp.init();
  __syncthreads();
  if (splitType == CRITERION::GINI) {
    giniGain<DataT, IdxT>(shist, sbins, sp, col, range_len, nbins, nclasses);
  } else {
    entropyGain<DataT, IdxT>(shist, sbins, sp, col, range_len, nbins, nclasses);
  }
  __syncthreads();
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
__global__ void computeSplitRegressionKernel(
  DataT* pred, DataT* pred2, DataT* pred2P, IdxT* count, IdxT nbins,
  IdxT max_depth, IdxT min_samples_split, IdxT max_leaves,
  Input<DataT, LabelT, IdxT> input, const Node<DataT, LabelT, IdxT>* nodes,
  IdxT colStart, int* done_count, int* mutex, const IdxT* n_leaves,
  Split<DataT, IdxT>* splits, void* workspace, CRITERION splitType, IdxT treeid,
  uint64_t seed) {
  extern __shared__ char smem[];
  IdxT nid = blockIdx.z;
  auto node = nodes[nid];
  auto range_start = node.start;
  auto range_len = node.count;
  if (leafBasedOnParams<DataT, IdxT>(node.depth, max_depth, min_samples_split,
                                     max_leaves, n_leaves, range_len)) {
    return;
  }
  auto end = range_start + range_len;
  auto len = nbins * 2;
  auto* spred = alignPointer<DataT>(smem);
  auto* scount = alignPointer<int>(spred + len);
  auto* sbins = alignPointer<DataT>(scount + nbins);

  // used only for MAE criterion
  auto* spred2 = alignPointer<DataT>(sbins + nbins);
  auto* spred2P = alignPointer<DataT>(spred2 + len);
  auto* spredP = alignPointer<DataT>(spred2P + nbins);
  auto* sDone = alignPointer<int>(spredP + nbins);
  IdxT stride = blockDim.x * gridDim.x;
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  // auto col = input.colids[colStart + blockIdx.y];
  int colIndex = blockIdx.y;
  auto col = selectFeature(treeid, nid, colIndex, seed, input.N);
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    spred[i] = DataT(0.0);
  }
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    scount[i] = 0;
    // printf("indexing from sbins: %p  to %p, sizeof: %d (spred: %p)\n", sbins,
    //        &sbins[i], (int)sizeof(DataT*), spred);
    sbins[i] = input.quantiles[col * nbins + i];
  }
  __syncthreads();
  auto coloffset = col * input.M;

  // compute prediction averages for all bins in shared mem
  for (auto i = range_start + tid; i < end; i += stride) {
    auto row = input.rowids[i];
    auto d = input.data[row + coloffset];
    auto label = input.labels[row];
    for (IdxT b = 0; b < nbins; ++b) {
      auto isRight = d > sbins[b];  // no divergence
      auto offset = isRight * nbins + b;
      atomicAdd(spred + offset, label);
      if (!isRight) atomicAdd(scount + b, 1);
    }
  }
  __syncthreads();
  // update the corresponding global location
  auto gcOffset = ((nid * gridDim.y) + blockIdx.y) * nbins;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    atomicAdd(count + gcOffset + i, scount[i]);
  }
  auto gOffset = gcOffset * 2;
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    atomicAdd(pred + gOffset + i, spred[i]);
  }
  __threadfence();  // for commit guarantee
  __syncthreads();
  // for MAE computation, we'd need a 2nd pass over data :(
  if (splitType == CRITERION::MAE) {
    // wait until all blockIdx.x's are done
    MLCommon::GridSync gs(workspace, MLCommon::SyncType::ACROSS_X, false);
    gs.sync();
    // now, compute the mean value to be used for MAE update
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      scount[i] = count[gcOffset + i];
      spred2P[i] = DataT(0.0);
    }
    for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
      spred[i] = pred[gOffset + i];
      spred2[i] = DataT(0.0);
    }
    __syncthreads();
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      spredP[i] = spred[i] + spred[i + nbins];
    }
    __syncthreads();
    auto invlen = DataT(1.0) / range_len;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      auto cnt_l = DataT(scount[i]);
      auto cnt_r = DataT(range_len - scount[i]);
      spred[i] /= cnt_l;
      spred[i + nbins] /= cnt_r;
      spredP[i] *= invlen;
    }
    __syncthreads();
    // 2nd pass over data to compute partial MAE's across blockIdx.x's
    for (auto i = range_start + tid; i < end; i += stride) {
      auto row = input.rowids[i];
      auto d = input.data[row + coloffset];
      auto label = input.labels[row];
      for (IdxT b = 0; b < nbins; ++b) {
        auto isRight = d > sbins[b];  // no divergence
        auto offset = isRight * nbins + b;
        auto diff = label - (isRight ? spred[nbins + b] : spred[b]);
        atomicAdd(spred2 + offset, raft::myAbs(diff));
        atomicAdd(spred2P + b, raft::myAbs(label - spredP[b]));
      }
    }
    __syncthreads();
    // update the corresponding global location
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      atomicAdd(pred2P + gcOffset + i, spred2P[i]);
    }
    for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
      atomicAdd(pred2 + gOffset + i, spred2[i]);
    }
    __threadfence();  // for commit guarantee
    __syncthreads();
  }
  // last threadblock will go ahead and compute the best split
  bool last = true;
  if (gridDim.x > 1) {
    last = MLCommon::signalDone(done_count + nid * gridDim.y + blockIdx.y,
                                gridDim.x, blockIdx.x == 0, sDone);
  }

  if (!last) return;
  // last block computes the final gain
  Split<DataT, IdxT> sp;
  sp.init();
  if (splitType == CRITERION::MSE) {
    for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
      spred[i] = pred[gOffset + i];
    }
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      scount[i] = count[gcOffset + i];
    }
    __syncthreads();
    mseGain(spred, scount, sbins, sp, col, range_len, nbins);
  } else {
    for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
      spred2[i] = pred2[gOffset + i];
    }
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      spred2P[i] = pred2P[gcOffset + i];
    }
    __syncthreads();
    maeGain(spred2, spred2P, scount, sbins, sp, col, range_len, nbins);
  }
  __syncthreads();
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}

}  // namespace DecisionTree
}  // namespace ML
