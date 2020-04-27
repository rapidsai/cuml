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
struct MSEGain {
  static HDI T exec(const T parent_best_metric, const unsigned int total,
                    const unsigned int left, const unsigned int right,
                    const T parent_mean, T &mean_left, T &mean_right,
                    T &mse_left, T &mse_right) {
    mean_right /= right;
    mean_left /= left;
    mse_left = mean_left;
    mse_right = mean_right;
    T left_impurity = ((float)left / total) * mean_left * mean_left;
    T right_impurity = ((float)right / total) * mean_right * mean_right;
    T temp = left_impurity + right_impurity - (parent_mean * parent_mean);
    return temp;
  }
  static HDI T exec(const unsigned int total, const unsigned int left,
                    const unsigned int right, const T parent_mean, T &mean_left,
                    T &mean_right) {
    mean_right /= right;
    mean_left /= left;
    T left_impurity = ((float)left / total) * mean_left * mean_left;
    T right_impurity = ((float)right / total) * mean_right * mean_right;
    T temp = left_impurity + right_impurity - (parent_mean * parent_mean);
    return temp;
  }
};

template <typename T>
struct MAEGain {
  static HDI T exec(const T parent_best_metric, const unsigned int total,
                    const unsigned int left, const unsigned int right,
                    const T parent_mean, T &mean_left, T &mean_right,
                    T &mae_left, T &mae_right) {
    mean_left /= left;
    mean_right /= right;
    mae_left /= left;
    mae_right /= right;
    T left_impurity = (left * 1.0 / total) * mae_left;
    T right_impurity = (right * 1.0 / total) * mae_right;
    return (parent_best_metric - (left_impurity + right_impurity));
  }
  static HDI T exec(const T parent_mae, const T mae_left, const T mae_right,
                    const unsigned int left, const unsigned int right,
                    const unsigned int total) {
    T left_impurity = (left * 1.0 / total) * mae_left;
    T right_impurity = (right * 1.0 / total) * mae_right;
    return (parent_mae - (left_impurity + right_impurity));
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

//This is device version of best split in case, used when more than 512 nodes.
template <typename T, typename Gain>
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

      float info_gain = (float)Gain::exec(
        parent_metric, parent_count, tmp_lnrows, tmp_rnrows, parent_mean,
        tmp_meanleft, tmp_meanright, tmp_mse_left, tmp_mse_right);

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

//This is the best bin finder at block level for each column using one pass MSE
template <typename T>
DI GainIdxPair bin_info_gain_regression_mse(const T sum_parent,
                                            const T *sum_left,
                                            const unsigned int *count_right,
                                            const int count, const int nbins) {
  GainIdxPair tid_pair;
  tid_pair.gain = 0.0;
  tid_pair.idx = -1;
  for (int tid = threadIdx.x; tid < nbins; tid += blockDim.x) {
    unsigned int right = count_right[tid];
    unsigned int left = count - right;
    if ((right != 0) && (left != 0)) {
      T mean_left = sum_left[tid];
      T mean_right = sum_parent - mean_left;
      T mean_parent = sum_parent / count;
      float info_gain = (float)MSEGain<T>::exec(count, left, right, mean_parent,
                                                mean_left, mean_right);
      if (info_gain > tid_pair.gain) {
        tid_pair.gain = info_gain;
        tid_pair.idx = tid;
      }
    }
  }
  return tid_pair;
}

//This is the best bin finder at block level for each column using two pass MAE
template <typename T>
DI GainIdxPair bin_info_gain_regression_mae(const T mae_sum_parent,
                                            const T *mae_sum_left,
                                            const T *mae_sum_right,
                                            const unsigned int *count_right,
                                            const int count, const int nbins) {
  GainIdxPair tid_pair;
  tid_pair.gain = 0.0;
  tid_pair.idx = -1;
  for (int tid = threadIdx.x; tid < nbins; tid += blockDim.x) {
    unsigned int right = count_right[tid];
    unsigned int left = count - right;
    if ((right != 0) && (left != 0)) {
      T mae_left = mae_sum_left[tid] / (T)left;
      T mae_right = mae_sum_right[tid] / (T)right;
      T mae_parent = mae_sum_parent / (T)count;
      float info_gain = (float)MAEGain<T>::exec(mae_parent, mae_left, mae_right,
                                                left, right, count);
      if (info_gain > tid_pair.gain) {
        tid_pair.gain = info_gain;
        tid_pair.idx = tid;
      }
    }
  }
  return tid_pair;
}

//One pass best split using MSE
template <typename T, typename QuestionType, int TPB>
__global__ void best_split_gather_regression_mse_kernel(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart, const T *__restrict__ question_ptr,
  const unsigned int *__restrict__ g_nodestart,
  const unsigned int *__restrict__ samplelist, const int n_nodes,
  const int nbins, const int nrows, const int Ncols, const int ncols_sampled,
  const size_t treesz, const float min_impurity_split,
  SparseTreeNode<T, T> *d_sparsenodes, int *d_nodelist) {
  //shmemhist_parent[n_unique_labels]
  extern __shared__ char shmem_mse_gather[];
  T *shmean_left = (T *)(shmem_mse_gather);
  unsigned int *shcount_right = (unsigned int *)(shmean_left + nbins);
  __shared__ T mean_parent;
  __shared__ GainIdxPair shmem_pair;
  __shared__ int shmem_col;
  typedef cub::BlockReduce<GainIdxPair, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int colstart_local = -1;
  int colid;
  T local_label;
  unsigned int dataid;
  unsigned int nodestart = g_nodestart[blockIdx.x];
  unsigned int count = g_nodestart[blockIdx.x + 1] - nodestart;
  if (colstart != nullptr) colstart_local = colstart[blockIdx.x];

  //Compute parent histograms
  if (threadIdx.x == 0) {
    mean_parent = 0.0;
    shmem_pair.gain = 0.0f;
    shmem_pair.idx = -1;
    shmem_col = -1;
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    dataid = samplelist[nodestart + tid];
    local_label = labels[dataid];
    atomicAdd(&mean_parent, local_label);
  }

  //Loop over cols
  for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
    colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled, colcnt,
                          blockIdx.x);
    for (int tid = threadIdx.x; tid < 2 * nbins; tid += blockDim.x) {
      if (tid < nbins)
        shmean_left[tid] = (T)0.0;
      else
        shcount_right[tid - nbins] = 0;
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
        if (local_data <= question(binid)) {
          atomicAdd(&shmean_left[binid], local_label);
        } else {
          atomicAdd(&shcount_right[binid], 1);
        }
      }
    }
    __syncthreads();
    GainIdxPair bin_pair = bin_info_gain_regression_mse<T>(
      mean_parent, shmean_left, shcount_right, count, nbins);
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
    SparseTreeNode<T, T> localnode;
    if ((shmem_col != -1) && (shmem_pair.gain > min_impurity_split)) {
      colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                            shmem_col, blockIdx.x);
      QuestionType question(question_ptr, colid, shmem_col, n_nodes, blockIdx.x,
                            nbins);
      localnode.quesval = question(shmem_pair.idx);
      localnode.left_child_id = treesz + 2 * blockIdx.x;
    } else {
      colid = -1;
      localnode.prediction = mean_parent / count;
    }
    localnode.colid = colid;
    localnode.best_metric_val = mean_parent / count;
    d_sparsenodes[d_nodelist[blockIdx.x]] = localnode;
  }
}

//Same as above but fused with minmax mode, one pass min/max
// one pass MSE. total two pass.
template <typename T, typename E, int TPB>
__global__ void best_split_gather_regression_mse_minmax_kernel(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart,
  const unsigned int *__restrict__ g_nodestart,
  const unsigned int *__restrict__ samplelist, const int n_nodes,
  const int nbins, const int nrows, const int Ncols, const int ncols_sampled,
  const size_t treesz, const float min_impurity_split, const T init_min_val,
  SparseTreeNode<T, T> *d_sparsenodes, int *d_nodelist) {
  //shmemhist_parent[n_unique_labels]
  extern __shared__ char shmem_mse_minmax_gather[];
  T *shmean_left = (T *)(shmem_mse_minmax_gather);
  unsigned int *shcount_right = (unsigned int *)(shmean_left + nbins);
  __shared__ T mean_parent;
  __shared__ GainIdxPair shmem_pair;
  __shared__ int shmem_col;
  typedef cub::BlockReduce<GainIdxPair, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T shmem_min, shmem_max, best_min, best_delta;

  int colstart_local = -1;
  int colid;
  T local_label;
  unsigned int dataid;
  T local_data;
  unsigned int nodestart = g_nodestart[blockIdx.x];
  unsigned int count = g_nodestart[blockIdx.x + 1] - nodestart;
  if (colstart != nullptr) colstart_local = colstart[blockIdx.x];

  //Compute parent histograms
  if (threadIdx.x == 0) {
    mean_parent = 0.0;
    shmem_pair.gain = 0.0f;
    shmem_pair.idx = -1;
    shmem_col = -1;
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    dataid = samplelist[nodestart + tid];
    local_label = labels[dataid];
    atomicAdd(&mean_parent, local_label);
  }

  //Loop over cols
  for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
    if (threadIdx.x == 0) {
      *(E *)&shmem_min = MLCommon::Stats::encode(init_min_val);
      *(E *)&shmem_max = MLCommon::Stats::encode(-init_min_val);
    }
    colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled, colcnt,
                          blockIdx.x);
    for (int tid = threadIdx.x; tid < 2 * nbins; tid += blockDim.x) {
      if (tid < nbins)
        shmean_left[tid] = (T)0.0;
      else
        shcount_right[tid - nbins] = 0;
    }
    __syncthreads();

    //Compuet minmax oon independent data pass
    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      unsigned int dataid = samplelist[nodestart + tid];
      local_data = data[dataid + colid * nrows];
      MLCommon::Stats::atomicMinBits<T, E>(&shmem_min, local_data);
      MLCommon::Stats::atomicMaxBits<T, E>(&shmem_max, local_data);
    }
    __syncthreads();

    T threadmin = MLCommon::Stats::decode(*(E *)&shmem_min);
    T delta =
      (MLCommon::Stats::decode(*(E *)&shmem_max) - threadmin) / (nbins + 1);

    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      dataid = get_samplelist(samplelist, dataid, nodestart, tid, count);
      local_data = get_data(data, local_data, dataid + colid * nrows, count);
      local_label = get_label(labels, local_label, dataid, count);
#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        if (local_data <= threadmin + delta * (binid + 1)) {
          atomicAdd(&shmean_left[binid], local_label);
        } else {
          atomicAdd(&shcount_right[binid], 1);
        }
      }
    }
    __syncthreads();
    GainIdxPair bin_pair = bin_info_gain_regression_mse<T>(
      mean_parent, shmean_left, shcount_right, count, nbins);
    GainIdxPair best_bin_pair =
      BlockReduce(temp_storage).Reduce(bin_pair, ReducePair<cub::Max>());
    __syncthreads();

    if ((best_bin_pair.gain > shmem_pair.gain) && (threadIdx.x == 0)) {
      shmem_pair = best_bin_pair;
      shmem_col = colcnt;
      best_min = threadmin;
      best_delta = delta;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    SparseTreeNode<T, T> localnode;
    if ((shmem_col != -1) && (shmem_pair.gain > min_impurity_split)) {
      colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                            shmem_col, blockIdx.x);
      localnode.quesval = best_min + (shmem_pair.idx + 1) * best_delta;
      localnode.left_child_id = treesz + 2 * blockIdx.x;
    } else {
      colid = -1;
      localnode.prediction = mean_parent / count;
    }
    localnode.colid = colid;
    localnode.best_metric_val = mean_parent / count;
    d_sparsenodes[d_nodelist[blockIdx.x]] = localnode;
  }
}

//A light weight implementation of the best split kernel for last level,
// when all nodes are to be leafed out. works for all algo all split criteria
template <typename T>
__global__ void make_leaf_gather_regression_kernel(
  const T *__restrict__ labels, const unsigned int *__restrict__ g_nodestart,
  const unsigned int *__restrict__ samplelist,
  SparseTreeNode<T, T> *d_sparsenodes, int *d_nodelist) {
  __shared__ T mean_parent;
  unsigned int nodestart = g_nodestart[blockIdx.x];
  unsigned int count = g_nodestart[blockIdx.x + 1] - nodestart;

  //Compute parent histograms
  mean_parent = 0.0f;
  __syncthreads();

  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    unsigned int dataid = samplelist[nodestart + tid];
    T local_label = labels[dataid];
    atomicAdd(&mean_parent, local_label);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    SparseTreeNode<T, T> localnode;
    localnode.prediction = mean_parent / count;
    localnode.colid = -1;
    localnode.best_metric_val = mean_parent / count;
    d_sparsenodes[d_nodelist[blockIdx.x]] = localnode;
  }
}

//Gather kernel for MAE. We need this different as MAE needs to be multipass
// One pass for mean and one pass for MAE
template <typename T, typename QuestionType, int TPB>
__global__ void best_split_gather_regression_mae_kernel(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart, const T *__restrict__ question_ptr,
  const unsigned int *__restrict__ g_nodestart,
  const unsigned int *__restrict__ samplelist, const int n_nodes,
  const int nbins, const int nrows, const int Ncols, const int ncols_sampled,
  const size_t treesz, const float min_impurity_split,
  SparseTreeNode<T, T> *d_sparsenodes, int *d_nodelist) {
  extern __shared__ char shmem_mae_gather[];
  T *shmean_left = (T *)shmem_mae_gather;
  T *shmae_left = (T *)(shmean_left + nbins);
  T *shmae_right = (T *)(shmae_left + nbins);
  unsigned int *shcount_right = (unsigned int *)(shmae_right + nbins);
  __shared__ T mean_parent, mae_parent;
  __shared__ GainIdxPair shmem_pair;
  __shared__ int shmem_col;
  typedef cub::BlockReduce<GainIdxPair, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int colstart_local = -1;
  int colid;
  T local_label;
  T local_data;
  unsigned int dataid;
  unsigned int nodestart = g_nodestart[blockIdx.x];
  unsigned int count = g_nodestart[blockIdx.x + 1] - nodestart;
  if (colstart != nullptr) colstart_local = colstart[blockIdx.x];

  //Compute parent histograms
  if (threadIdx.x == 0) {
    mean_parent = 0.0;
    shmem_pair.gain = 0.0f;
    shmem_pair.idx = -1;
    shmem_col = -1;
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    dataid = samplelist[nodestart + tid];
    local_label = labels[dataid];
    atomicAdd(&mean_parent, local_label);
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    dataid = get_samplelist(samplelist, dataid, nodestart, tid, count);
    local_label = get_label(labels, local_label, dataid, count);
    T value = (mean_parent / count) - local_label;
    atomicAdd(&mae_parent, MLCommon::myAbs(value));
  }
  //Loop over cols
  for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
    colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled, colcnt,
                          blockIdx.x);
    for (int tid = threadIdx.x; tid < 2 * nbins; tid += blockDim.x) {
      if (tid < nbins) {
        shmean_left[tid] = (T)0;
        shmae_left[tid] = (T)0;
      } else {
        shcount_right[tid - nbins] = 0;
        shmae_right[tid - nbins] = (T)0;
      }
    }
    QuestionType question(question_ptr, colid, colcnt, n_nodes, blockIdx.x,
                          nbins);
    __syncthreads();
    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      dataid = get_samplelist(samplelist, dataid, nodestart, tid, count);
      local_data = data[dataid + colid * nrows];
      local_label = get_label(labels, local_label, dataid, count);
#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        if (local_data <= question(binid)) {
          atomicAdd(&shmean_left[binid], local_label);
        } else {
          atomicAdd(&shcount_right[binid], 1);
        }
      }
    }
    __syncthreads();
    //second data pass is needed for MAE
    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      dataid = get_samplelist(samplelist, dataid, nodestart, tid, count);
      local_data = get_data(data, local_data, dataid + colid * nrows, count);
      local_label = get_label(labels, local_label, dataid, count);
#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        if (local_data <= question(binid)) {
          T value =
            (shmean_left[binid] / (count - shcount_right[binid])) - local_label;
          atomicAdd(&shmae_left[binid], MLCommon::myAbs(value));
        } else {
          T value =
            ((mean_parent - shmean_left[binid]) / shcount_right[binid]) -
            local_label;
          atomicAdd(&shmae_right[binid], MLCommon::myAbs(value));
        }
      }
    }
    __syncthreads();
    GainIdxPair bin_pair = bin_info_gain_regression_mae<T>(
      mae_parent, shmae_left, shmae_right, shcount_right, count, nbins);
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
    SparseTreeNode<T, T> localnode;
    if ((shmem_col != -1) && (shmem_pair.gain > min_impurity_split)) {
      colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                            shmem_col, blockIdx.x);
      QuestionType question(question_ptr, colid, shmem_col, n_nodes, blockIdx.x,
                            nbins);
      localnode.quesval = question(shmem_pair.idx);
      localnode.left_child_id = treesz + 2 * blockIdx.x;
    } else {
      colid = -1;
      localnode.prediction = mean_parent / count;
    }
    localnode.colid = colid;
    localnode.best_metric_val = mae_parent / count;
    d_sparsenodes[d_nodelist[blockIdx.x]] = localnode;
  }
}

//Same as above but fused with minmax mode, one pass min/max
// one pass Mean. one pass MAE. total three passes.
template <typename T, typename E, int TPB>
__global__ void best_split_gather_regression_mae_minmax_kernel(
  const T *__restrict__ data, const T *__restrict__ labels,
  const unsigned int *__restrict__ colids,
  const unsigned int *__restrict__ colstart,
  const unsigned int *__restrict__ g_nodestart,
  const unsigned int *__restrict__ samplelist, const int n_nodes,
  const int nbins, const int nrows, const int Ncols, const int ncols_sampled,
  const size_t treesz, const float min_impurity_split, const T init_min_val,
  SparseTreeNode<T, T> *d_sparsenodes, int *d_nodelist) {
  //shmemhist_parent[n_unique_labels]
  extern __shared__ char shmem_mae_minmax_gather[];
  T *shmean_left = (T *)shmem_mae_minmax_gather;
  T *shmae_left = (T *)(shmean_left + nbins);
  T *shmae_right = (T *)(shmae_left + nbins);
  unsigned int *shcount_right = (unsigned int *)(shmae_right + nbins);
  __shared__ T mean_parent, mae_parent;
  __shared__ GainIdxPair shmem_pair;
  __shared__ int shmem_col;
  typedef cub::BlockReduce<GainIdxPair, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T shmem_min, shmem_max, best_min, best_delta;

  int colstart_local = -1;
  int colid;
  T local_label;
  unsigned int dataid;
  T local_data;
  unsigned int nodestart = g_nodestart[blockIdx.x];
  unsigned int count = g_nodestart[blockIdx.x + 1] - nodestart;
  if (colstart != nullptr) colstart_local = colstart[blockIdx.x];

  //Compute parent histograms
  if (threadIdx.x == 0) {
    mean_parent = 0.0;
    mae_parent = 0.0;
    shmem_pair.gain = 0.0f;
    shmem_pair.idx = -1;
    shmem_col = -1;
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    dataid = samplelist[nodestart + tid];
    local_label = labels[dataid];
    atomicAdd(&mean_parent, local_label);
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    dataid = get_samplelist(samplelist, dataid, nodestart, tid, count);
    local_label = get_label(labels, local_label, dataid, count);
    T value = (mean_parent / count) - local_label;
    atomicAdd(&mae_parent, MLCommon::myAbs(value));
  }

  //Loop over cols
  for (unsigned int colcnt = 0; colcnt < ncols_sampled; colcnt++) {
    if (threadIdx.x == 0) {
      *(E *)&shmem_min = MLCommon::Stats::encode(init_min_val);
      *(E *)&shmem_max = MLCommon::Stats::encode(-init_min_val);
    }
    colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled, colcnt,
                          blockIdx.x);
    for (int tid = threadIdx.x; tid < 2 * nbins; tid += blockDim.x) {
      if (tid < nbins) {
        shmean_left[tid] = (T)0;
        shmae_left[tid] = (T)0;
      } else {
        shcount_right[tid - nbins] = 0;
        shmae_right[tid - nbins] = (T)0;
      }
    }
    __syncthreads();

    //Compuet minmax on independent data pass
    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      unsigned int dataid = samplelist[nodestart + tid];
      local_data = data[dataid + colid * nrows];
      MLCommon::Stats::atomicMinBits<T, E>(&shmem_min, local_data);
      MLCommon::Stats::atomicMaxBits<T, E>(&shmem_max, local_data);
    }
    __syncthreads();

    T threadmin = MLCommon::Stats::decode(*(E *)&shmem_min);
    T delta =
      (MLCommon::Stats::decode(*(E *)&shmem_max) - threadmin) / (nbins + 1);

    //Second pass for Mean
    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      dataid = get_samplelist(samplelist, dataid, nodestart, tid, count);
      local_data = get_data(data, local_data, dataid + colid * nrows, count);
      local_label = get_label(labels, local_label, dataid, count);
#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        if (local_data <= threadmin + delta * (binid + 1)) {
          atomicAdd(&shmean_left[binid], local_label);
        } else {
          atomicAdd(&shcount_right[binid], 1);
        }
      }
    }
    __syncthreads();
    //Third pass needed for MAE
    for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
      dataid = get_samplelist(samplelist, dataid, nodestart, tid, count);
      local_data = get_data(data, local_data, dataid + colid * nrows, count);
      local_label = get_label(labels, local_label, dataid, count);
#pragma unroll(8)
      for (unsigned int binid = 0; binid < nbins; binid++) {
        if (local_data <= threadmin + delta * (binid + 1)) {
          T value =
            (shmean_left[binid] / (count - shcount_right[binid])) - local_label;
          atomicAdd(&shmae_left[binid], MLCommon::myAbs(value));
        } else {
          T value =
            ((mean_parent - shmean_left[binid]) / shcount_right[binid]) -
            local_label;
          atomicAdd(&shmae_right[binid], MLCommon::myAbs(value));
        }
      }
    }
    __syncthreads();

    GainIdxPair bin_pair = bin_info_gain_regression_mae<T>(
      mae_parent, shmae_left, shmae_right, shcount_right, count, nbins);
    GainIdxPair best_bin_pair =
      BlockReduce(temp_storage).Reduce(bin_pair, ReducePair<cub::Max>());
    __syncthreads();

    if ((best_bin_pair.gain > shmem_pair.gain) && (threadIdx.x == 0)) {
      shmem_pair = best_bin_pair;
      shmem_col = colcnt;
      best_min = threadmin;
      best_delta = delta;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    SparseTreeNode<T, T> localnode;
    if ((shmem_col != -1) && (shmem_pair.gain > min_impurity_split)) {
      colid = get_column_id(colids, colstart_local, Ncols, ncols_sampled,
                            shmem_col, blockIdx.x);
      localnode.quesval = best_min + (shmem_pair.idx + 1) * best_delta;
      localnode.left_child_id = treesz + 2 * blockIdx.x;
    } else {
      colid = -1;
      localnode.prediction = mean_parent / count;
    }
    localnode.colid = colid;
    localnode.best_metric_val = mae_parent / count;
    d_sparsenodes[d_nodelist[blockIdx.x]] = localnode;
  }
}
