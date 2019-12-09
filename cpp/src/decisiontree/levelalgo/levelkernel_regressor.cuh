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
struct MSEImpurity {
  static HDI T exec(const unsigned int total, const unsigned int left,
                    const unsigned int right, const T parent_mean,
                    const T sumleft, const T sumsq_left, const T sumsq_right) {
    T temp = sumleft / total;
    T sumright = (parent_mean * total) - sumleft;
    T left_impurity = (sumsq_left / total) - (total / left) * temp * temp;
    temp = sumright / total;
    T right_impurity = (sumsq_right / total) - (total / right) * temp * temp;
    return (left_impurity + right_impurity);
  }
};

template <typename T>
struct MAEImpurity {
  static HDI T exec(const unsigned int total, const unsigned int left,
                    const unsigned int right, const T parent_mean,
                    const T sumleft, const T mae_left, const T mae_right) {
    T left_impurity = (left * 1.0 / total) * (mae_left / left);
    T right_impurity = (right * 1.0 / total) * (mae_right / right);
    return (left_impurity + right_impurity);
  }
};

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
__global__ void get_pred_kernel(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ flags,
  const unsigned int *__restrict__ sample_cnt,
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart, const int nrows, const int Ncols,
  const int ncols_sampled, const int nbins, const int n_nodes,
  const T *__restrict__ question_ptr, T *predout, unsigned int *countout) {
  extern __shared__ char shmem_pred_kernel[];
  T *shmempred = (T *)shmem_pred_kernel;
  unsigned int *shmemcount =
    (unsigned int *)(&shmem_pred_kernel[nbins * n_nodes * sizeof(T)]);
  unsigned int local_flag = LEAF;
  T local_label;
  int local_cnt;
  int colstart_local = -1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int colid;
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
    for (unsigned int i = threadIdx.x; i < nbins * n_nodes; i += blockDim.x) {
      shmempred[i] = (T)0;
      shmemcount[i] = 0;
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

//This kernel computes sum left , count left and sqared sum both left and right,
//for all colls, all bins and all nodes at a given level
template <typename T, typename QuestionType>
__global__ void get_mse_pred_kernel(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ flags,
  const unsigned int *__restrict__ sample_cnt,
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart, const int nrows, const int Ncols,
  const int ncols_sampled, const int nbins, const int n_nodes,
  const T *__restrict__ question_ptr, T *predout, unsigned int *countout,
  T *mseout) {
  extern __shared__ char shmem_mse_pred_kernel[];
  T *shmem_predout = (T *)(shmem_mse_pred_kernel);
  T *shmem_mse = (T *)(shmem_mse_pred_kernel + n_nodes * nbins * sizeof(T));
  unsigned int *shmem_countout =
    (unsigned int *)(shmem_mse_pred_kernel + 3 * n_nodes * nbins * sizeof(T));

  unsigned int local_flag = LEAF;
  T local_label;
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
    unsigned int coloff = colcnt * nbins * n_nodes;
    for (unsigned int i = threadIdx.x; i < nbins * n_nodes; i += blockDim.x) {
      shmem_predout[i] = (T)0;
      shmem_countout[i] = 0;
    }

    for (unsigned int i = threadIdx.x; i < 2 * nbins * n_nodes;
         i += blockDim.x) {
      shmem_mse[i] = (T)0;
    }
    __syncthreads();

    //Check if leaf
    if (local_flag != LEAF) {
      T local_data = data[tid + colid * nrows];
      QuestionType question(question_ptr, colid, colcnt, n_nodes, local_flag,
                            nbins);

#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        unsigned int nodeoff = local_flag * nbins;
        if (local_data <= question(binid)) {
          atomicAdd(&shmem_countout[nodeoff + binid], local_cnt);
          atomicAdd(&shmem_predout[nodeoff + binid], local_label);
          atomicAdd(&shmem_mse[2 * (nodeoff + binid)],
                    local_label * local_label);
        } else {
          atomicAdd(&shmem_mse[2 * (nodeoff + binid) + 1],
                    local_label * local_label);
        }
      }
    }

    __syncthreads();
    for (unsigned int i = threadIdx.x; i < 2 * nbins * n_nodes;
         i += blockDim.x) {
      atomicAdd(&mseout[2 * coloff + i], shmem_mse[i]);
    }
    for (unsigned int i = threadIdx.x; i < nbins * n_nodes; i += blockDim.x) {
      atomicAdd(&predout[coloff + i], shmem_predout[i]);
      atomicAdd(&countout[coloff + i], shmem_countout[i]);
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
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart, const int nrows, const int Ncols,
  const int ncols_sampled, const int nbins, const int n_nodes,
  const T *__restrict__ question_ptr, const T *__restrict__ parentpred,
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
  unsigned int colid;
  int colstart_local = -1;
  if (tid < nrows) {
    local_flag = flags[tid];
  }

  if (local_flag != LEAF) {
    parent_count = parentcount[local_flag];
    parent_pred = parentpred[local_flag];
    local_label = labels[tid];
    local_cnt = sample_cnt[tid];
    if (colstart != nullptr) colstart_local = colstart[local_flag];
  }

  for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
    if (local_flag != LEAF) {
      colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                            colcnt, local_flag);
    }
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
      QuestionType question(question_ptr, colid, colcnt, n_nodes, local_flag,
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
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart, const int nrows, const int Ncols,
  const int ncols_sampled, const int nbins, const int n_nodes,
  const T *__restrict__ question_ptr, T *predout, unsigned int *countout) {
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
      int colstart_local = -1;
      if (colstart != nullptr) colstart_local = colstart[local_flag];

      for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
        unsigned int colid = get_column_id(colids, colstart_local, Ncols,
                                           ncols_sampled, colcnt, local_flag);
        unsigned int coloffset = colcnt * nbins * n_nodes;
        T local_data = data[tid + colid * nrows];
        QuestionType question(question_ptr, colid, colcnt, n_nodes, local_flag,
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
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart, const int nrows, const int Ncols,
  const int ncols_sampled, const int nbins, const int n_nodes,
  const T *__restrict__ question_ptr, const T *__restrict__ parentpred,
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
    if (local_flag != LEAF) {
      local_label = labels[tid];
      local_cnt = sample_cnt[tid];
      parent_count = parentcount[local_flag];
      parent_pred = parentpred[local_flag];
      int colstart_local = -1;
      if (colstart != nullptr) colstart_local = colstart[local_flag];

      for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
        unsigned int colid = get_column_id(colids, colstart_local, Ncols,
                                           ncols_sampled, colcnt, local_flag);
        unsigned int coloff = colcnt * nbins * n_nodes;
        T local_data = data[tid + colid * nrows];
        QuestionType question(question_ptr, colid, colcnt, n_nodes, local_flag,
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

//This kernel computes sum left , count left and sqared sum both left and right,
//for all colls, all bins and all nodes at a given level, a global mem version
template <typename T, typename QuestionType>
__global__ void get_mse_pred_kernel_global(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ flags,
  const unsigned int *__restrict__ sample_cnt,
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart, const int nrows, const int Ncols,
  const int ncols_sampled, const int nbins, const int n_nodes,
  const T *__restrict__ question_ptr, T *predout, unsigned int *countout,
  T *mseout) {
  unsigned int local_flag = LEAF;
  T local_label;
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
    unsigned int coloff = colcnt * nbins * n_nodes;

    //Check if leaf
    if (local_flag != LEAF) {
      T local_data = data[tid + colid * nrows];
      QuestionType question(question_ptr, colid, colcnt, n_nodes, local_flag,
                            nbins);

#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        unsigned int nodeoff = local_flag * nbins;
        if (local_data <= question(binid)) {
          atomicAdd(&countout[coloff + nodeoff + binid], local_cnt);
          atomicAdd(&predout[coloff + nodeoff + binid], local_label);
          atomicAdd(&mseout[2 * (coloff + nodeoff + binid)],
                    local_label * local_label);
        } else {
          atomicAdd(&mseout[2 * (coloff + nodeoff + binid) + 1],
                    local_label * local_label);
        }
      }
    }
  }
}

//This is device version of best split in case, used when more than 512 nodes.
template <typename T, typename Impurity>
__global__ void get_best_split_regression_kernel(
  const T *__restrict__ mseout, const T *__restrict__ predout,
  const unsigned int *__restrict__ count, const T *__restrict__ parentmean,
  const unsigned int *__restrict__ parentcount,
  const T *__restrict__ parentmetric, const int nbins, const int ncols_sampled,
  const int n_nodes, const int min_rpn, float *outgain, int *best_col_id,
  int *best_bin_id, T *child_mean, unsigned int *child_count,
  T *child_best_metric) {
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
    tid_pair.idx = -1;
    for (int id = threadIdx.x; id < nbins * ncols_sampled; id += blockDim.x) {
      int coloffset = ((int)(id / nbins)) * nbins * n_nodes;
      int binoffset = id % nbins;
      int threadoffset = coloffset + binoffset + nodeoffset;
      unsigned int tmp_lnrows = count[threadoffset];
      unsigned int tmp_rnrows = parent_count - tmp_lnrows;
      unsigned int totalrows = tmp_lnrows + tmp_rnrows;
      if (tmp_lnrows == 0 || tmp_rnrows == 0 || totalrows < min_rpn) continue;
      T tmp_meanleft = predout[threadoffset];
      T tmp_meanright = parent_mean * parent_count - tmp_meanleft;
      T tmp_mse_left = mseout[2 * threadoffset];
      T tmp_mse_right = mseout[2 * threadoffset + 1];

      T impurity =
        Impurity::exec(parent_count, tmp_lnrows, tmp_rnrows, parent_mean,
                       tmp_meanleft, tmp_mse_left, tmp_mse_right);

      tmp_meanleft /= tmp_lnrows;
      tmp_meanright /= tmp_rnrows;
      tmp_mse_left /= tmp_lnrows;
      tmp_mse_right /= tmp_rnrows;
      float info_gain = (float)(parent_metric - impurity);

      if (info_gain > tid_pair.gain) {
        tid_pair.gain = info_gain;
        tid_pair.idx = id;
      }
    }
    __syncthreads();
    GainIdxPair ans =
      BlockReduce(temp_storage).Reduce(tid_pair, ReducePair<cub::Max>());

    if (threadIdx.x == 0 && ans.idx != -1) {
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
