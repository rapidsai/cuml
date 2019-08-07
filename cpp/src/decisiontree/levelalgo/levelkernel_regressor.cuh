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

template <typename T>
__global__ void pred_kernel_level(const T *__restrict__ labels,
                                  const unsigned int *__restrict__ sample_cnt,
                                  const int nrows, T *predout,
                                  unsigned int *countout) {
  int threadid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ T shmempred;
  __shared__ unsigned int shmemcnt;
  if (threadIdx.x == 0) {
    shmempred = 0;
    shmemcnt = 0;
  }
  __syncthreads();

  for (int tid = threadid; tid < nrows; tid += blockDim.x * gridDim.x) {
    T label = labels[tid];
    unsigned int count = sample_cnt[tid];
    atomicAdd(&shmemcnt, count);
    atomicAdd(&shmempred, label * count);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(predout, shmempred);
    atomicAdd(countout, shmemcnt);
  }
  return;
}

template <typename T, typename F>
__global__ void mse_kernel_level(const T *__restrict__ labels,
                                 const unsigned int *__restrict__ sample_cnt,
                                 const int nrows, const T *predout,
                                 const unsigned int *count, T *mseout) {
  int threadid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ T shmemmse;
  if (threadIdx.x == 0) shmemmse = 0;
  __syncthreads();

  T mean = predout[0] / count[0];
  for (int tid = threadid; tid < nrows; tid += blockDim.x * gridDim.x) {
    T label = labels[tid];
    unsigned int local_count = sample_cnt[tid];
    T value = F::exec(label - mean);
    atomicAdd(&shmemmse, local_count * value);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(mseout, shmemmse);
  }
  return;
}
//This kernel computes predictions and count for all colls, all bins and all nodes at a given level
template <typename T, typename QuestionType>
__global__ void get_pred_kernel(const T *__restrict__ data,
                                const T *__restrict__ labels,
                                const unsigned int *__restrict__ flags,
                                const unsigned int *__restrict__ sample_cnt,
                                const unsigned int *__restrict__ colids,
                                const int nrows, const int ncols,
                                const int nbins, const int n_nodes,
                                const T *__restrict__ question_ptr, T *predout,
                                unsigned int *countout) {
  extern __shared__ char shmem_pred_kernel[];
  T *shmempred = (T *)shmem_pred_kernel;
  unsigned int *shmemcount =
    (unsigned int *)(&shmem_pred_kernel[nbins * n_nodes * sizeof(T)]);
  unsigned int local_flag = LEAF;
  T local_label;
  int local_cnt;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < nrows) {
    local_flag = flags[tid];
    local_label = labels[tid];
    local_cnt = sample_cnt[tid];
  }

  for (unsigned int colcnt = 0; colcnt < ncols; colcnt++) {
    unsigned int colid = colids[colcnt];
    for (unsigned int i = threadIdx.x; i < nbins * n_nodes; i += blockDim.x) {
      shmempred[i] = (T)0;
      shmemcount[i] = 0;
    }
    __syncthreads();

    //Check if leaf
    if (local_flag != LEAF) {
      T local_data = data[tid + colid * nrows];
      QuestionType question(question_ptr, colids, colcnt, n_nodes, local_flag,
                            nbins);

#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        if (local_data <= question(binid)) {
          unsigned int nodeoff = local_flag * nbins;
          atomicAdd(&shmempred[nodeoff + binid], local_label * local_cnt);
          atomicAdd(&shmemcount[nodeoff + binid], local_cnt);
        }
      }
    }

    __syncthreads();
    for (unsigned int i = threadIdx.x; i < nbins * n_nodes; i += blockDim.x) {
      unsigned int offset = colcnt * nbins * n_nodes;
      atomicAdd(&predout[offset + i], shmempred[i]);
      atomicAdd(&countout[offset + i], shmemcount[i]);
    }
    __syncthreads();
  }
}

//This kernel computes mse/mae for all colls, all bins and all nodes at a given level
template <typename T, typename F, typename QuestionType>
__global__ void get_mse_kernel(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ flags,
  const unsigned int *__restrict__ sample_cnt,
  const unsigned int *__restrict__ colids, const int nrows, const int ncols,
  const int nbins, const int n_nodes, const T *__restrict__ question_ptr,
  const T *__restrict__ parentpred,
  const unsigned int *__restrict__ parentcount, const T *__restrict__ predout,
  const unsigned int *__restrict__ countout, T *mseout) {
  extern __shared__ char shmem_mse_kernel[];
  T *shmem_predout = (T *)(shmem_mse_kernel);
  T *shmem_mse = (T *)(shmem_mse_kernel + n_nodes * nbins * sizeof(T));
  unsigned int *shmem_countout =
    (unsigned int *)(shmem_mse_kernel + 3 * n_nodes * nbins * sizeof(T));

  unsigned int local_flag = LEAF;
  T local_label;
  int local_cnt;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  T parent_pred;
  unsigned int parent_count;

  if (tid < nrows) {
    local_flag = flags[tid];
  }

  if (local_flag != LEAF) {
    parent_count = parentcount[local_flag];
    parent_pred = parentpred[local_flag];
    local_label = labels[tid];
    local_cnt = sample_cnt[tid];
  }

  for (unsigned int colcnt = 0; colcnt < ncols; colcnt++) {
    unsigned int colid = colids[colcnt];
    unsigned int coloff = colcnt * nbins * n_nodes;
    for (unsigned int i = threadIdx.x; i < nbins * n_nodes; i += blockDim.x) {
      shmem_predout[i] = predout[i + coloff];
      shmem_countout[i] = countout[i + coloff];
    }

    for (unsigned int i = threadIdx.x; i < 2 * nbins * n_nodes;
         i += blockDim.x) {
      shmem_mse[i] = (T)0;
    }
    __syncthreads();

    //Check if leaf
    if (local_flag != LEAF) {
      T local_data = data[tid + colid * nrows];
      QuestionType question(question_ptr, colids, colcnt, n_nodes, local_flag,
                            nbins);

#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        unsigned int nodeoff = local_flag * nbins;
        T local_pred = shmem_predout[nodeoff + binid];
        unsigned int local_count = shmem_countout[nodeoff + binid];
        if (local_data <= question(binid)) {
          T leftmean = local_pred / local_count;
          atomicAdd(&shmem_mse[2 * (nodeoff + binid)],
                    local_cnt * F::exec(local_label - leftmean));
        } else {
          T rightmean = parent_pred * parent_count - local_pred;
          rightmean = rightmean / (parent_count - local_count);
          atomicAdd(&shmem_mse[2 * (nodeoff + binid) + 1],
                    local_cnt * F::exec(local_label - rightmean));
        }
      }
    }

    __syncthreads();
    for (unsigned int i = threadIdx.x; i < 2 * nbins * n_nodes;
         i += blockDim.x) {
      atomicAdd(&mseout[2 * coloff + i], shmem_mse[i]);
    }
    __syncthreads();
  }
}

//This kernel computes predictions and count for all colls, all bins and all nodes at a given level
//This is when nodes dont fit anymore in shared memory.
template <typename T, typename QuestionType>
__global__ void get_pred_kernel_global(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ flags,
  const unsigned int *__restrict__ sample_cnt,
  const unsigned int *__restrict__ colids, const int nrows, const int ncols,
  const int nbins, const int n_nodes, const T *__restrict__ question_ptr,
  T *predout, unsigned int *countout) {
  unsigned int local_flag = LEAF;
  T local_label;
  int local_cnt;
  int threadid = threadIdx.x + blockIdx.x * blockDim.x;

  for (int tid = threadid; tid < nrows; tid += blockDim.x * gridDim.x) {
    local_flag = flags[tid];
    //Check if leaf
    if (local_flag != LEAF) {
      local_label = labels[tid];
      local_cnt = sample_cnt[tid];

      for (unsigned int colcnt = 0; colcnt < ncols; colcnt++) {
        unsigned int colid = colids[colcnt];
        unsigned int coloffset = colcnt * nbins * n_nodes;
        T local_data = data[tid + colid * nrows];
        QuestionType question(question_ptr, colids, colcnt, n_nodes, local_flag,
                              nbins);

#pragma unroll(8)
        for (unsigned int binid = 0; binid < nbins; binid++) {
          if (local_data <= question(binid)) {
            unsigned int nodeoff = local_flag * nbins;
            atomicAdd(&predout[coloffset + nodeoff + binid],
                      local_label * local_cnt);
            atomicAdd(&countout[coloffset + nodeoff + binid], local_cnt);
          }
        }
      }
    }
  }
}

//This kernel computes mse/mae for all colls, all bins and all nodes at a given level
// This is when nodes dont fit in shared memory
template <typename T, typename F, typename QuestionType>
__global__ void get_mse_kernel_global(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ flags,
  const unsigned int *__restrict__ sample_cnt,
  const unsigned int *__restrict__ colids, const int nrows, const int ncols,
  const int nbins, const int n_nodes, const T *__restrict__ question_ptr,
  const T *__restrict__ parentpred,
  const unsigned int *__restrict__ parentcount, const T *__restrict__ predout,
  const unsigned int *__restrict__ countout, T *mseout) {
  unsigned int local_flag = LEAF;
  T local_label;
  int local_cnt;
  int threadid = threadIdx.x + blockIdx.x * blockDim.x;
  T parent_pred;
  unsigned int parent_count;

  for (int tid = threadid; tid < nrows; tid += gridDim.x * blockDim.x) {
    local_flag = flags[tid];
    local_label = labels[tid];
    local_cnt = sample_cnt[tid];

    if (local_flag != LEAF) {
      parent_count = parentcount[local_flag];
      parent_pred = parentpred[local_flag];

      for (unsigned int colcnt = 0; colcnt < ncols; colcnt++) {
        unsigned int colid = colids[colcnt];
        unsigned int coloff = colcnt * nbins * n_nodes;
        T local_data = data[tid + colid * nrows];
        QuestionType question(question_ptr, colids, colcnt, n_nodes, local_flag,
                              nbins);

#pragma unroll(8)
        for (unsigned int binid = 0; binid < nbins; binid++) {
          unsigned int nodeoff = local_flag * nbins;
          T local_pred = predout[coloff + nodeoff + binid];
          unsigned int local_count = countout[coloff + nodeoff + binid];
          if (local_data <= question(binid)) {
            T leftmean = local_pred / local_count;
            atomicAdd(&mseout[2 * (coloff + nodeoff + binid)],
                      local_cnt * F::exec(local_label - leftmean));
          } else {
            T rightmean = parent_pred * parent_count - local_pred;
            rightmean = rightmean / (parent_count - local_count);
            atomicAdd(&mseout[2 * (coloff + nodeoff + binid) + 1],
                      local_cnt * F::exec(local_label - rightmean));
          }
        }
      }
    }
  }
}
//This is device version of best split in case, used when more than 512 nodes.
template <typename T>
__global__ void get_best_split_regression_kernel(
  const T *__restrict__ mseout, const T *__restrict__ predout,
  const unsigned int *__restrict__ count, const T *__restrict__ parentmean,
  const unsigned int *__restrict__ parentcount,
  const T *__restrict__ parentmetric, const unsigned int *__restrict__ colids,
  const int nbins, const int ncols, const int n_nodes, const int min_rpn,
  float *outgain, int *best_col_id, int *best_bin_id, T *child_mean,
  unsigned int *child_count, T *child_best_metric) {
  typedef cub::BlockReduce<GainIdxPair, 64> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (unsigned int nodeid = blockIdx.x; nodeid < n_nodes;
       nodeid += gridDim.x) {
    T parent_mean = parentmean[nodeid];
    unsigned int parent_count = parentcount[nodeid];
    T parent_metric = parentmetric[nodeid];
    int nodeoffset = nodeid * nbins;
    GainIdxPair tid_pair;
    tid_pair.gain = 0.0;
    tid_pair.idx = 0;
    for (int id = threadIdx.x; id < nbins * ncols; id += blockDim.x) {
      int coloffset = ((int)(id / nbins)) * nbins * n_nodes;
      int binoffset = id % nbins;
      int threadoffset = coloffset + binoffset + nodeoffset;
      unsigned int tmp_lnrows = count[threadoffset];
      unsigned int tmp_rnrows = parent_count - tmp_lnrows;
      unsigned int totalrows = tmp_lnrows + tmp_rnrows;
      if (tmp_lnrows == 0 || tmp_rnrows == 0 || totalrows < min_rpn) continue;
      T tmp_meanleft = predout[threadoffset];
      T tmp_meanright = parent_mean * parent_count - tmp_meanleft;
      tmp_meanleft /= tmp_lnrows;
      tmp_meanright /= tmp_rnrows;
      T tmp_mse_left = mseout[2 * threadoffset] / tmp_lnrows;
      T tmp_mse_right = mseout[2 * threadoffset + 1] / tmp_rnrows;

      T impurity = (tmp_lnrows * 1.0 / totalrows) * tmp_mse_left +
                   (tmp_rnrows * 1.0 / totalrows) * tmp_mse_right;
      float info_gain = (float)(parent_metric - impurity);

      if (info_gain > tid_pair.gain) {
        tid_pair.gain = info_gain;
        tid_pair.idx = id;
      }
    }
    __syncthreads();
    GainIdxPair ans =
      BlockReduce(temp_storage).Reduce(tid_pair, ReducePair<cub::Max>());

    if (threadIdx.x == 0) {
      outgain[nodeid] = ans.gain;
      best_col_id[nodeid] = (int)(ans.idx / nbins);
      best_bin_id[nodeid] = ans.idx % nbins;
      int coloffset = ((int)(ans.idx / nbins)) * nbins * n_nodes;
      int binoffset = ans.idx % nbins;
      int threadoffset = coloffset + binoffset + nodeoffset;
      if (ans.idx != -1) {
        unsigned int tmp_lnrows = count[threadoffset];
        child_count[2 * nodeid] = tmp_lnrows;
        unsigned int tmp_rnrows = parent_count - tmp_lnrows;
        child_count[2 * nodeid + 1] = tmp_rnrows;
        T tmp_meanleft = predout[threadoffset];
        child_mean[2 * nodeid] = tmp_meanleft / tmp_lnrows;
        child_mean[2 * nodeid + 1] =
          (parent_mean * parent_count - tmp_meanleft) / tmp_rnrows;
        child_best_metric[2 * nodeid] = mseout[2 * threadoffset] / tmp_lnrows;
        child_best_metric[2 * nodeid + 1] =
          mseout[2 * threadoffset + 1] / tmp_rnrows;
      }
    }
  }
}
