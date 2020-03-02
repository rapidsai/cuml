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
#include "common_kernel.cuh"
#include "cub/cub.cuh"

__global__ void sample_count_histogram_kernel(
  const int* __restrict__ labels, const unsigned int* __restrict__ sample_cnt,
  const int nrows, const int nmax, int* histout) {
  int threadid = threadIdx.x + blockIdx.x * blockDim.x;
  extern __shared__ unsigned int shmemhist[];
  for (int tid = threadIdx.x; tid < nmax; tid += blockDim.x) {
    shmemhist[tid] = 0;
  }

  __syncthreads();

  for (int tid = threadid; tid < nrows; tid += blockDim.x * gridDim.x) {
    int label = labels[tid];
    int count = sample_cnt[tid];
    atomicAdd(&shmemhist[label], count);
  }

  __syncthreads();

  for (int tid = threadIdx.x; tid < nmax; tid += blockDim.x) {
    atomicAdd(&histout[tid], shmemhist[tid]);
  }
  return;
}

//This kernel does histograms for all bins, all cols and all nodes at a given level
template <typename T, typename QuestionType>
__global__ void get_hist_kernel(
  const T* __restrict__ data, const int* __restrict__ labels,
  const unsigned int* __restrict__ flags,
  const unsigned int* __restrict__ sample_cnt,
  const unsigned int* __restrict__ colids,
  const unsigned int* __restrict__ colstart, const int nrows, const int Ncols,
  const int ncols_sampled, const int n_unique_labels, const int nbins,
  const int n_nodes, const T* __restrict__ question_ptr,
  unsigned int* histout) {
  extern __shared__ unsigned int shmemhist[];
  unsigned int local_flag = LEAF;
  int local_label = -1;
  int local_cnt;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int colid;
  int colstart_local = -1;
  if (tid < nrows) {
    local_flag = flags[tid];
  }
  if (local_flag != LEAF) {
    local_label = labels[tid];
    local_cnt = sample_cnt[tid];
    if (colstart != nullptr) colstart_local = colstart[local_flag];
  }
  for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
    if (local_flag != LEAF) {
      colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                            colcnt, local_flag);
    }
    for (unsigned int i = threadIdx.x; i < nbins * n_nodes * n_unique_labels;
         i += blockDim.x) {
      shmemhist[i] = 0;
    }
    __syncthreads();

    //Check if leaf
    if (local_flag != LEAF) {
      T local_data = data[tid + colid * nrows];
      QuestionType question(question_ptr, colid, colcnt, n_nodes, local_flag,
                            nbins);
#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        if (local_data <= question(binid)) {
          unsigned int nodeoff = local_flag * nbins * n_unique_labels;
          atomicAdd(&shmemhist[nodeoff + binid * n_unique_labels + local_label],
                    local_cnt);
        }
      }
    }

    __syncthreads();
    for (unsigned int i = threadIdx.x; i < nbins * n_nodes * n_unique_labels;
         i += blockDim.x) {
      unsigned int offset = colcnt * nbins * n_nodes * n_unique_labels;
      atomicAdd(&histout[offset + i], shmemhist[i]);
    }
    __syncthreads();
  }
}

/*This kernel does histograms for all bins, all cols and all nodes at a given level
 *when nodes cannot fit in shared memory. We use direct global atomics;
 *as this will be faster than shared memory loop due to reduced conjetion for atomics
 */
template <typename T, typename QuestionType>
__global__ void get_hist_kernel_global(
  const T* __restrict__ data, const int* __restrict__ labels,
  const unsigned int* __restrict__ flags,
  const unsigned int* __restrict__ sample_cnt,
  const unsigned int* __restrict__ colids,
  const unsigned int* __restrict__ colstart, const int nrows, const int Ncols,
  const int ncols_sampled, const int n_unique_labels, const int nbins,
  const int n_nodes, const T* __restrict__ question_ptr,
  unsigned int* histout) {
  unsigned int local_flag;
  int local_label;
  int local_cnt;
  int threadid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int tid = threadid; tid < nrows; tid += gridDim.x * blockDim.x) {
    local_flag = flags[tid];
    if (local_flag != LEAF) {
      local_label = labels[tid];
      local_cnt = sample_cnt[tid];
      int colstart_local = -1;
      if (colstart != nullptr) colstart_local = colstart[local_flag];

      for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
        unsigned int colid = get_column_id(colids, colstart_local, Ncols,
                                           ncols_sampled, colcnt, local_flag);
        T local_data = data[tid + colid * nrows];
        //Loop over nbins
        QuestionType question(question_ptr, colid, colcnt, n_nodes, local_flag,
                              nbins);

#pragma unroll(8)
        for (unsigned int binid = 0; binid < nbins; binid++) {
          if (local_data <= question(binid)) {
            unsigned int coloff = colcnt * nbins * n_nodes * n_unique_labels;
            unsigned int nodeoff = local_flag * nbins * n_unique_labels;
            atomicAdd(&histout[coloff + nodeoff + binid * n_unique_labels +
                               local_label],
                      local_cnt);
          }
        }
      }
    }
  }
}

struct GiniDevFunctor {
  static DI float exec(unsigned int* hist, int nrows, int n_unique_labels) {
    float gval = 1.0;
    for (int i = 0; i < n_unique_labels; i++) {
      float prob = ((float)hist[i]) / nrows;
      gval -= prob * prob;
    }
    return gval;
  }
  static DI void execshared(const unsigned int* hist, float* metric,
                            const int nrows, const int n_unique_labels) {
    auto& tid = threadIdx.x;
    if (tid == 0) metric[0] = 1.0;
    __syncthreads();
    if (tid < n_unique_labels) {
      float prob = ((float)hist[tid]) / nrows;
      prob = -1 * prob * prob;
      atomicAdd(metric, prob);
    }
    __syncthreads();
  }
};

struct EntropyDevFunctor {
  static DI float exec(unsigned int* hist, int nrows, int n_unique_labels) {
    float eval = 0.0;
    for (int i = 0; i < n_unique_labels; i++) {
      if (hist[i] != 0) {
        float prob = ((float)hist[i]) / nrows;
        eval += prob * logf(prob);
      }
    }
    return (-1 * eval);
  }
  static DI void execshared(const unsigned int* hist, float* metric,
                            const int nrows, const int n_unique_labels) {
    auto& tid = threadIdx.x;
    if (tid == 0) metric[0] = 0.0;
    __syncthreads();
    if (tid < n_unique_labels) {
      if (hist[tid] != 0) {
        float prob = ((float)hist[tid]) / nrows;
        prob = -1 * prob * logf(prob);
        atomicAdd(metric, prob);
      }
    }
    __syncthreads();
  }
};
//This is device equialent of best split finding reduction.
//Only kicks in when number of node is more than 512. otherwise we use CPU.
template <typename T, typename F>
__global__ void get_best_split_classification_kernel(
  const unsigned int* __restrict__ hist,
  const unsigned int* __restrict__ parent_hist,
  const T* __restrict__ parent_metric, const int nbins, const int ncols_sampled,
  const int n_nodes, const int n_unique_labels, const int min_rpn,
  float* outgain, int* best_col_id, int* best_bin_id, unsigned int* child_hist,
  T* child_best_metric) {
  extern __shared__ unsigned int shmem_split_eval[];
  __shared__ int best_nrows[2];
  __shared__ GainIdxPair shared_pair;
  typedef cub::BlockReduce<GainIdxPair, 64> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  unsigned int* tmp_histleft = &shmem_split_eval[threadIdx.x * n_unique_labels];
  unsigned int* tmp_histright =
    &shmem_split_eval[threadIdx.x * n_unique_labels +
                      blockDim.x * n_unique_labels];
  unsigned int* best_split_hist =
    &shmem_split_eval[2 * n_unique_labels * blockDim.x];
  unsigned int* parent_hist_local =
    &shmem_split_eval[2 * n_unique_labels * (blockDim.x + 1)];

  for (unsigned int nodeid = blockIdx.x; nodeid < n_nodes;
       nodeid += gridDim.x) {
    if (threadIdx.x < 2) {
      best_nrows[threadIdx.x] = 0;
    }

    int nodeoffset = nodeid * nbins * n_unique_labels;
    float parent_metric_local = parent_metric[nodeid];

    for (int j = threadIdx.x; j < n_unique_labels; j += blockDim.x) {
      parent_hist_local[j] = parent_hist[nodeid * n_unique_labels + j];
    }

    __syncthreads();

    GainIdxPair tid_pair;
    tid_pair.gain = 0.0;
    tid_pair.idx = -1;
    for (int id = threadIdx.x; id < nbins * ncols_sampled; id += blockDim.x) {
      int coloffset = ((int)(id / nbins)) * nbins * n_unique_labels * n_nodes;
      int binoffset = (id % nbins) * n_unique_labels;
      int tmp_lnrows = 0;
      int tmp_rnrows = 0;
      for (int j = 0; j < n_unique_labels; j++) {
        tmp_histleft[j] = hist[coloffset + binoffset + nodeoffset + j];
        tmp_lnrows += tmp_histleft[j];
        tmp_histright[j] = parent_hist_local[j] - tmp_histleft[j];
        tmp_rnrows += tmp_histright[j];
      }

      int totalrows = tmp_lnrows + tmp_rnrows;
      if (tmp_lnrows == 0 || tmp_rnrows == 0 || totalrows < min_rpn) continue;

      float tmp_gini_left = F::exec(tmp_histleft, tmp_lnrows, n_unique_labels);
      float tmp_gini_right =
        F::exec(tmp_histright, tmp_rnrows, n_unique_labels);

      float impurity = (tmp_lnrows * 1.0f / totalrows) * tmp_gini_left +
                       (tmp_rnrows * 1.0f / totalrows) * tmp_gini_right;
      float info_gain = parent_metric_local - impurity;
      if (info_gain > tid_pair.gain) {
        tid_pair.gain = info_gain;
        tid_pair.idx = id;
      }
    }
    __syncthreads();
    GainIdxPair ans =
      BlockReduce(temp_storage).Reduce(tid_pair, ReducePair<cub::Max>());

    if (threadIdx.x == 0) {
      shared_pair = ans;
    }
    __syncthreads();
    ans = shared_pair;

    if (ans.idx != -1) {
      if (threadIdx.x == (blockDim.x - 1)) {
        outgain[nodeid] = ans.gain;
        best_col_id[nodeid] = (int)(ans.idx / nbins);
        best_bin_id[nodeid] = ans.idx % nbins;
      }

      int coloffset =
        ((int)(ans.idx / nbins)) * nbins * n_unique_labels * n_nodes;
      int binoffset = (ans.idx % nbins) * n_unique_labels;

      for (int j = threadIdx.x; j < n_unique_labels; j += blockDim.x) {
        unsigned int val_left = hist[coloffset + binoffset + nodeoffset + j];
        unsigned int val_right = parent_hist_local[j] - val_left;
        best_split_hist[j] = val_left;
        atomicAdd(&best_nrows[0], val_left);
        best_split_hist[j + n_unique_labels] = val_right;
        atomicAdd(&best_nrows[1], val_right);
      }
      __syncthreads();

      for (int j = threadIdx.x; j < 2 * n_unique_labels; j += blockDim.x) {
        child_hist[2 * n_unique_labels * nodeid + j] = best_split_hist[j];
      }

      if (threadIdx.x < 2) {
        child_best_metric[2 * nodeid + threadIdx.x] =
          F::exec(&best_split_hist[threadIdx.x * n_unique_labels],
                  best_nrows[threadIdx.x], n_unique_labels);
      }
    }
  }
}

template <typename F>
DI GainIdxPair bin_info_gain_classification(
  const unsigned int* shmemhist_parent, const float* parent_metric,
  unsigned int* shmemhist_left, const int nsamples, const int nbins,
  const int n_unique_labels) {
  GainIdxPair tid_pair;
  tid_pair.gain = 0.0;
  tid_pair.idx = -1;
  for (int tid = threadIdx.x; tid < nbins; tid += blockDim.x) {
    int nrows_left = 0;
    unsigned int* shmemhist = shmemhist_left + tid * n_unique_labels;
    for (int i = 0; i < n_unique_labels; i++) {
      nrows_left += shmemhist[i];
    }
    if ((nrows_left != nsamples) && (nrows_left != 0)) {
      int nrows_right = nsamples - nrows_left;
      float left_metric = F::exec(shmemhist, nrows_left, n_unique_labels);
      for (int i = 0; i < n_unique_labels; i++) {
        shmemhist[i] = shmemhist_parent[i] - shmemhist[i];
      }
      float right_metric = F::exec(shmemhist, nrows_right, n_unique_labels);
      float impurity = ((nrows_left * 1.0f) / nsamples) * left_metric +
                       ((nrows_right * 1.0f) / nsamples) * right_metric;
      float info_gain = parent_metric[0] - impurity;
      if (info_gain > tid_pair.gain) {
        tid_pair.gain = info_gain;
        tid_pair.idx = tid;
      }
    }
  }
  return tid_pair;
}

template <typename T, typename QuestionType, typename FDEV, int TPB>
__global__ void best_split_gather_classification_kernel(
  const T* __restrict__ data, const int* __restrict__ labels,
  const unsigned int* __restrict__ colids,
  const unsigned int* __restrict__ colstart, const T* __restrict__ question_ptr,
  const unsigned int* __restrict__ g_nodestart,
  const unsigned int* __restrict__ samplelist, const int n_nodes,
  const int n_unique_labels, const int nbins, const int nrows, const int Ncols,
  const int ncols_sampled, const size_t treesz, const float min_impurity_split,
  SparseTreeNode<T, int>* d_sparsenodes, int* d_nodelist) {
  //shmemhist_parent[n_unique_labels]
  extern __shared__ unsigned int shmemhist_parent[];
  __shared__ GainIdxPair shmem_pair;
  __shared__ int shmem_col;
  __shared__ float parent_metric;
  typedef cub::BlockReduce<GainIdxPair, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  //shmemhist_left[n_unique_labels*nbins]
  unsigned int* shmemhist_left = shmemhist_parent + n_unique_labels;

  int colstart_local = -1;
  int colid;
  int local_label;
  unsigned int dataid;
  unsigned int nodestart = g_nodestart[blockIdx.x];
  unsigned int count = g_nodestart[blockIdx.x + 1] - nodestart;
  if (colstart != nullptr) colstart_local = colstart[blockIdx.x];

  //Compute parent histograms
  for (int i = threadIdx.x; i < n_unique_labels; i += blockDim.x) {
    shmemhist_parent[i] = 0;
  }
  if (threadIdx.x == 0) {
    shmem_pair.gain = 0.0f;
    shmem_pair.idx = -1;
    shmem_col = -1;
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    dataid = samplelist[nodestart + tid];
    local_label = labels[dataid];
    atomicAdd(&shmemhist_parent[local_label], 1);
  }
  FDEV::execshared(shmemhist_parent, &parent_metric, count, n_unique_labels);
  //Loop over cols
  for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
    colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled, colcnt,
                          blockIdx.x);
    for (int i = threadIdx.x; i < nbins * n_unique_labels; i += blockDim.x) {
      shmemhist_left[i] = 0;
    }
    QuestionType question(question_ptr, colid, colcnt, n_nodes, blockIdx.x,
                          nbins);
    __syncthreads();
    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      dataid = get_samplelist(samplelist, dataid, nodestart, tid, count);
      T local_data = data[dataid + colid * nrows];
      local_label = get_label(labels, local_label, dataid, count);
#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        int histid = binid * n_unique_labels + local_label;
        if (local_data <= question(binid)) {
          atomicAdd(&shmemhist_left[histid], 1);
        }
      }
    }
    __syncthreads();
    GainIdxPair bin_pair = bin_info_gain_classification<FDEV>(
      shmemhist_parent, &parent_metric, shmemhist_left, count, nbins,
      n_unique_labels);
    GainIdxPair best_bin_pair =
      BlockReduce(temp_storage).Reduce(bin_pair, ReducePair<cub::Max>());
    __syncthreads();

    if ((best_bin_pair.gain > shmem_pair.gain) && (threadIdx.x == 0)) {
      shmem_pair = best_bin_pair;
      shmem_col = colcnt;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    SparseTreeNode<T, int> localnode;
    if ((shmem_col != -1) && (shmem_pair.gain > min_impurity_split)) {
      colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                            shmem_col, blockIdx.x);
      QuestionType question(question_ptr, colid, shmem_col, n_nodes, blockIdx.x,
                            nbins);
      localnode.quesval = question(shmem_pair.idx);
      localnode.left_child_id = treesz + 2 * blockIdx.x;
    } else {
      colid = -1;
      localnode.prediction =
        get_class_hist_shared(shmemhist_parent, n_unique_labels);
    }
    localnode.colid = colid;
    localnode.best_metric_val = parent_metric;
    d_sparsenodes[d_nodelist[blockIdx.x]] = localnode;
  }
}

//The same as above but fused minmax at block level
template <typename T, typename E, typename FDEV, int TPB>
__global__ void best_split_gather_classification_minmax_kernel(
  const T* __restrict__ data, const int* __restrict__ labels,
  const unsigned int* __restrict__ colids,
  const unsigned int* __restrict__ colstart,
  const unsigned int* __restrict__ g_nodestart,
  const unsigned int* __restrict__ samplelist, const int n_nodes,
  const int n_unique_labels, const int nbins, const int nrows, const int Ncols,
  const int ncols_sampled, const size_t treesz, const float min_impurity_split,
  const T init_min_val, SparseTreeNode<T, int>* d_sparsenodes,
  int* d_nodelist) {
  //shmemhist_parent[n_unique_labels]
  extern __shared__ unsigned int shmemhist_parent[];
  __shared__ GainIdxPair shmem_pair;
  __shared__ int shmem_col;
  __shared__ float parent_metric;
  typedef cub::BlockReduce<GainIdxPair, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T shmem_min, shmem_max, best_min, best_delta;
  //shmemhist_left[n_unique_labels*nbins]
  unsigned int* shmemhist_left = shmemhist_parent + n_unique_labels;

  int colstart_local = -1;
  int colid;
  int local_label;
  unsigned int dataid;
  T local_data;
  unsigned int nodestart = g_nodestart[blockIdx.x];
  unsigned int count = g_nodestart[blockIdx.x + 1] - nodestart;
  if (colstart != nullptr) colstart_local = colstart[blockIdx.x];

  //Compute parent histograms
  for (int i = threadIdx.x; i < n_unique_labels; i += blockDim.x) {
    shmemhist_parent[i] = 0;
  }
  if (threadIdx.x == 0) {
    shmem_pair.gain = 0.0f;
    shmem_pair.idx = -1;
    shmem_col = -1;
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    dataid = samplelist[nodestart + tid];
    local_label = labels[dataid];
    atomicAdd(&shmemhist_parent[local_label], 1);
  }
  FDEV::execshared(shmemhist_parent, &parent_metric, count, n_unique_labels);
  //Loop over cols
  for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
    if (threadIdx.x == 0) {
      *(E*)&shmem_min = MLCommon::Stats::encode(init_min_val);
      *(E*)&shmem_max = MLCommon::Stats::encode(-init_min_val);
    }
    colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled, colcnt,
                          blockIdx.x);
    for (int i = threadIdx.x; i < nbins * n_unique_labels; i += blockDim.x) {
      shmemhist_left[i] = 0;
    }
    __syncthreads();

    //compute min/max using independent data pass
    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      unsigned int dataid = samplelist[nodestart + tid];
      local_data = data[dataid + colid * nrows];
      MLCommon::Stats::atomicMinBits<T, E>(&shmem_min, local_data);
      MLCommon::Stats::atomicMaxBits<T, E>(&shmem_max, local_data);
    }
    __syncthreads();

    T threadmin = MLCommon::Stats::decode(*(E*)&shmem_min);
    T delta =
      (MLCommon::Stats::decode(*(E*)&shmem_max) - threadmin) / (nbins + 1);

    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      dataid = get_samplelist(samplelist, dataid, nodestart, tid, count);
      local_data = get_data(data, local_data, dataid + colid * nrows, count);
      local_label = get_label(labels, local_label, dataid, count);
#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        int histid = binid * n_unique_labels + local_label;
        if (local_data <= threadmin + delta * (binid + 1)) {
          atomicAdd(&shmemhist_left[histid], 1);
        }
      }
    }
    __syncthreads();
    GainIdxPair bin_pair = bin_info_gain_classification<FDEV>(
      shmemhist_parent, &parent_metric, shmemhist_left, count, nbins,
      n_unique_labels);
    GainIdxPair best_bin_pair =
      BlockReduce(temp_storage).Reduce(bin_pair, ReducePair<cub::Max>());
    __syncthreads();

    if ((best_bin_pair.gain > shmem_pair.gain)) {
      if (threadIdx.x == 0) {
        shmem_pair = best_bin_pair;
        shmem_col = colcnt;
        best_min = threadmin;
        best_delta = delta;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    SparseTreeNode<T, int> localnode;
    if ((shmem_col != -1) && (shmem_pair.gain > min_impurity_split)) {
      colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                            shmem_col, blockIdx.x);
      localnode.quesval = best_min + (shmem_pair.idx + 1) * best_delta;
      localnode.left_child_id = treesz + 2 * blockIdx.x;
    } else {
      colid = -1;
      localnode.prediction =
        get_class_hist_shared(shmemhist_parent, n_unique_labels);
    }
    localnode.colid = colid;
    localnode.best_metric_val = parent_metric;
    d_sparsenodes[d_nodelist[blockIdx.x]] = localnode;
  }
}

//A light weight implementation of the above kernel for last level,
// when all nodes are to be leafed out
template <typename T, typename FDEV>
__global__ void make_leaf_gather_classification_kernel(
  const int* __restrict__ labels, const unsigned int* __restrict__ g_nodestart,
  const unsigned int* __restrict__ samplelist, const int n_unique_labels,
  SparseTreeNode<T, int>* d_sparsenodes, int* d_nodelist) {
  __shared__ float parent_metric;
  //shmemhist_parent[n_unique_labels]
  extern __shared__ unsigned int shmemhist_parent[];
  unsigned int nodestart = g_nodestart[blockIdx.x];
  unsigned int count = g_nodestart[blockIdx.x + 1] - nodestart;

  //Compute parent histograms
  for (int i = threadIdx.x; i < n_unique_labels; i += blockDim.x) {
    shmemhist_parent[i] = 0;
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    unsigned int dataid = samplelist[nodestart + tid];
    int local_label = labels[dataid];
    atomicAdd(&shmemhist_parent[local_label], 1);
  }
  FDEV::execshared(shmemhist_parent, &parent_metric, count, n_unique_labels);
  __syncthreads();
  if (threadIdx.x == 0) {
    SparseTreeNode<T, int> localnode;
    localnode.prediction =
      get_class_hist_shared(shmemhist_parent, n_unique_labels);
    localnode.colid = -1;
    localnode.best_metric_val = parent_metric;
    d_sparsenodes[d_nodelist[blockIdx.x]] = localnode;
  }
}
