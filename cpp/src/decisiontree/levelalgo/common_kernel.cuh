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
#include "cuda_utils.h"
#define LEAF 0xFFFFFFFF
#define PUSHRIGHT 0x00000001
#include "stats/minmax.h"

template <typename T>
DI T get_data(const T* __restrict__ data, const T local_data,
              const unsigned int dataid, const unsigned int count) {
  if (count <= blockDim.x) {
    return local_data;
  } else {
    return data[dataid];
  }
}

DI unsigned int get_samplelist(const unsigned int* __restrict__ samplelist,
                               const unsigned int dataid,
                               const unsigned int nodestart, const int tid,
                               const unsigned int count) {
  if (count <= blockDim.x) {
    return dataid;
  } else {
    return samplelist[nodestart + tid];
  }
}

template <typename L>
DI L get_label(const L* __restrict__ labels, const L local_label,
               const unsigned int dataid, const unsigned int count) {
  if (count <= blockDim.x) {
    return local_label;
  } else {
    return labels[dataid];
  }
}
DI int get_class_hist_shared(unsigned int* node_hist,
                             const int n_unique_labels) {
  unsigned int maxval = node_hist[0];
  int classval = 0;
  for (int i = 1; i < n_unique_labels; i++) {
    if (node_hist[i] > maxval) {
      maxval = node_hist[i];
      classval = i;
    }
  }
  return classval;
}
__global__ void fill_all_leaf(unsigned int* flags, const int nrows) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nrows) {
    flags[tid] = LEAF;
  }
}
DI unsigned int get_column_id(const unsigned int* __restrict__ colids,
                              const int& colstart_local, const int& Ncols,
                              const int& ncols_sampled,
                              const unsigned int& colcnt,
                              const unsigned int& local_flag) {
  unsigned int col;
  if (colstart_local != -1) {
    col = colids[(colstart_local + colcnt) % Ncols];
  } else {
    col = colids[local_flag * ncols_sampled + colcnt];
  }
  return col;
}
template <typename T, typename E>
__global__ void minmax_init_kernel(T* minmax, const int len, const int n_nodes,
                                   const T init_val) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 2 * len) {
    bool ifmin = (((int)(tid / n_nodes) % 2) == 0);
    *(E*)&minmax[tid] = (ifmin) ? MLCommon::Stats::encode(init_val)
                                : MLCommon::Stats::encode(-init_val);
  }
}

template <typename T, typename E>
__global__ void minmax_decode_kernel(T* minmax, const int len) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 2 * len) {
    minmax[tid] = MLCommon::Stats::decode(*(E*)&minmax[tid]);
  }
}

//This kernel calculates minmax at node level
template <typename T, typename E>
__global__ void get_minmax_kernel(const T* __restrict__ data,
                                  const unsigned int* __restrict__ flags,
                                  const unsigned int* __restrict__ colids,
                                  const unsigned int* __restrict__ colstart,
                                  const int nrows, const int Ncols,
                                  const int ncols_sampled, const int n_nodes,
                                  T init_min_val, T* minmax) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int local_flag = LEAF;
  int colstart_local = -1;
  extern __shared__ char shared_mem_minmax[];
  T* shmem_minmax = (T*)shared_mem_minmax;
  if (tid < nrows) {
    local_flag = flags[tid];
  }
  if (local_flag != LEAF && colstart != nullptr) {
    colstart_local = colstart[local_flag];
  }
  for (int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
    for (int i = threadIdx.x; i < 2 * n_nodes; i += blockDim.x) {
      *(E*)&shmem_minmax[i] = (i < n_nodes)
                                ? MLCommon::Stats::encode(init_min_val)
                                : MLCommon::Stats::encode(-init_min_val);
    }

    __syncthreads();
    if (local_flag != LEAF) {
      int col = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                              colcnt, local_flag);
      T local_data = data[col * nrows + tid];
      if (!isnan(local_data)) {
        //Min max values are saved in shared memory and global memory as per the shuffled colids.
        MLCommon::Stats::atomicMinBits<T, E>(&shmem_minmax[local_flag],
                                             local_data);
        MLCommon::Stats::atomicMaxBits<T, E>(
          &shmem_minmax[local_flag + n_nodes], local_data);
      }
    }
    __syncthreads();

    //finally, perform global mem atomics
    for (int i = threadIdx.x; i < n_nodes; i += blockDim.x) {
      atomicMin((E*)&minmax[i + 2 * n_nodes * colcnt], *(E*)&shmem_minmax[i]);
      atomicMax((E*)&minmax[i + n_nodes + 2 * n_nodes * colcnt],
                *(E*)&shmem_minmax[i + n_nodes]);
    }
    __syncthreads();
  }
}

template <typename T, typename E>
__global__ void get_minmax_kernel_global(
  const T* __restrict__ data, const unsigned int* __restrict__ flags,
  const unsigned int* __restrict__ colids,
  const unsigned int* __restrict__ colstart, const int nrows, const int Ncols,
  const int ncols_sampled, const int n_nodes, T* minmax) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int local_flag = LEAF;
  if (tid < nrows) {
    local_flag = flags[tid];
    if (local_flag != LEAF) {
      int colstart_local = -1;
      if (colstart != nullptr) colstart_local = colstart[local_flag];
      for (int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
        int coloff = 2 * n_nodes * colcnt;
        int col = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                                colcnt, local_flag);
        T local_data = data[col * nrows + tid];
        if (!isnan(local_data)) {
          //Min max values are saved in shared memory and global memory as per the shuffled colids.
          MLCommon::Stats::atomicMinBits<T, E>(&minmax[coloff + local_flag],
                                               local_data);
          MLCommon::Stats::atomicMaxBits<T, E>(
            &minmax[coloff + n_nodes + local_flag], local_data);
        }
      }
    }
  }
}
//Setup how many times a sample is being used.
//This is due to bootstrap nature of Random Forest.
__global__ void setup_counts_kernel(unsigned int* sample_cnt,
                                    const unsigned int* __restrict__ rowids,
                                    const int n_sampled_rows) {
  int threadid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int tid = threadid; tid < n_sampled_rows;
       tid += blockDim.x * gridDim.x) {
    unsigned int stid = rowids[tid];
    atomicAdd(&sample_cnt[stid], 1);
  }
}
//This initializes the flags to 0x00000000. IF a sample is not used at all we Leaf out.
__global__ void setup_flags_kernel(const unsigned int* __restrict__ sample_cnt,
                                   unsigned int* flags, const int nrows) {
  int threadid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int tid = threadid; tid < nrows; tid += blockDim.x * gridDim.x) {
    unsigned int local_cnt = sample_cnt[tid];
    unsigned int local_flag = LEAF;
    if (local_cnt != 0) local_flag = 0x00000000;
    flags[tid] = local_flag;
  }
}

// This make actual split. A split is done using bits.
//Least significant Bit 0 means left and 1 means right.
//As a result a max depth of 32 is supported for now.
template <typename T, typename QuestionType>
__global__ void split_level_kernel(
  const T* __restrict__ data, const T* __restrict__ question_ptr,
  const unsigned int* __restrict__ colids,
  const unsigned int* __restrict__ colstart,
  const int* __restrict__ split_col_index,
  const int* __restrict__ split_bin_index, const int nrows, const int Ncols,
  const int ncols_sampled, const int nbins, const int n_nodes,
  const unsigned int* __restrict__ new_node_flags,
  unsigned int* __restrict__ flags) {
  unsigned int threadid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int local_flag = LEAF;

  for (int tid = threadid; tid < nrows; tid += gridDim.x * blockDim.x) {
    local_flag = flags[tid];

    if (local_flag != LEAF) {
      unsigned int local_leaf_flag = new_node_flags[local_flag];
      if (local_leaf_flag != LEAF) {
        int colidx = split_col_index[local_flag];
        int local_colstart = -1;
        if (colstart != nullptr) local_colstart = colstart[local_flag];
        int colid = get_column_id(colids, local_colstart, Ncols, ncols_sampled,
                                  colidx, local_flag);
        QuestionType question(question_ptr, colid, colidx, n_nodes, local_flag,
                              nbins);
        T quesval = question(split_bin_index[local_flag]);
        T local_data = data[colid * nrows + tid];
        //The inverse comparision here to push right instead of left
        if (local_data <= quesval) {
          local_flag = local_leaf_flag << 1;
        } else {
          local_flag = (local_leaf_flag << 1) | PUSHRIGHT;
        }
      } else {
        local_flag = LEAF;
      }
      flags[tid] = local_flag;
    }
  }
}

struct GainIdxPair {
  float gain;
  int idx;
};

template <typename KeyReduceOp>
struct ReducePair {
  KeyReduceOp op;
  DI ReducePair() {}
  DI ReducePair(KeyReduceOp op) : op(op) {}
  DI GainIdxPair operator()(const GainIdxPair& a, const GainIdxPair& b) {
    GainIdxPair retval;
    retval.gain = op(a.gain, b.gain);
    if (retval.gain == a.gain) {
      retval.idx = a.idx;
    } else {
      retval.idx = b.idx;
    }
    return retval;
  }
};

template <typename T>
struct QuantileQues {
  const T* __restrict__ quantile;
  DI QuantileQues(const T* __restrict__ quantile_ptr, const unsigned int colid,
                  const unsigned int colcnt, const int n_nodes,
                  const unsigned int nodeid, const int nbins)
    : quantile(quantile_ptr + colid * nbins) {}

  DI T operator()(const int binid) { return quantile[binid]; }
};

template <typename T>
struct MinMaxQues {
  T min, delta;
  DI MinMaxQues(const T* __restrict__ minmax_ptr, const unsigned int colid,
                const unsigned int colcnt, const int n_nodes,
                const unsigned int nodeid, const int nbins) {
    int off = colcnt * 2 * n_nodes + nodeid;
    min = minmax_ptr[off];
    delta = (minmax_ptr[off + n_nodes] - min) / nbins;
  }

  DI T operator()(const int binid) { return (min + (binid + 1) * delta); }
};

__global__ void fill_counts(const unsigned int* __restrict__ flagsptr,
                            const unsigned int* __restrict__ sample_cnt,
                            const int n_rows, unsigned int* nodecount) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_rows) {
    unsigned int nodeid = flagsptr[tid];
    if (nodeid != LEAF) {
      unsigned int count = sample_cnt[tid];
      atomicAdd(&nodecount[nodeid], count);
    }
  }
}

__global__ void build_list(const unsigned int* __restrict__ flagsptr,
                           const unsigned int* __restrict__ nodestart,
                           const int n_rows, unsigned int* nodecount,
                           unsigned int* samplelist) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_rows) {
    unsigned int nodeid = flagsptr[tid];
    if (nodeid != LEAF) {
      unsigned int start = nodestart[nodeid];
      unsigned int currcnt = atomicAdd(&nodecount[nodeid], 1);
      samplelist[start + currcnt] = tid;
    }
  }
}
template <typename T, typename L>
__global__ void split_nodes_compute_counts_kernel(
  const T* __restrict__ data,
  const SparseTreeNode<T, L>* __restrict__ d_sparsenodes,
  const unsigned int* __restrict__ nodestart,
  const unsigned int* __restrict__ samplelist, const int nrows,
  const int* __restrict__ nodelist, int* new_nodelist,
  unsigned int* samplecount, int* nodecounter, unsigned int* flagsptr) {
  __shared__ int currcnt;
  typedef cub::BlockReduce<int, 64> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char shmem[];
  SparseTreeNode<T, L>* localnode = (SparseTreeNode<T, L>*)shmem;
  if (threadIdx.x == 0) {
    localnode[0] = d_sparsenodes[nodelist[blockIdx.x]];
  }
  __syncthreads();
  int colid = localnode->colid;
  if (colid != -1) {
    unsigned int nstart = nodestart[blockIdx.x];
    unsigned int ncount = nodestart[blockIdx.x + 1] - nstart;
    if (threadIdx.x == 0) {
      currcnt = atomicAdd(nodecounter, 2);
    }
    __syncthreads();
    if (threadIdx.x < 2) {
      new_nodelist[currcnt + threadIdx.x] = 2 * blockIdx.x + threadIdx.x;
    }
    int tid_count = 0;
    T quesval = localnode->quesval;
    for (int tid = threadIdx.x; tid < ncount; tid += blockDim.x) {
      unsigned int dataid = samplelist[nstart + tid];
      if (data[colid * nrows + dataid] <= quesval) {
        tid_count++;
        flagsptr[dataid] = (unsigned int)(currcnt);
      } else {
        flagsptr[dataid] = (unsigned int)(currcnt + 1);
      }
    }
    int cnt_left = BlockReduce(temp_storage).Sum(tid_count);
    __syncthreads();
    if (threadIdx.x == 0) {
      samplecount[currcnt] = cnt_left;
      samplecount[currcnt + 1] = ncount - cnt_left;
    }
  }
}
