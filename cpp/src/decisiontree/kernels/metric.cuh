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
#include "metric_def.h"

template <class T>
void MetricQuestion<T>::set_question_fields(int cfg_bootcolumn, int cfg_column,
                                            int cfg_batch_id, int cfg_nbins,
                                            int cfg_ncols, T cfg_min, T cfg_max,
                                            T cfg_value) {
  bootstrapped_column = cfg_bootcolumn;
  original_column = cfg_column;
  batch_id = cfg_batch_id;
  min = cfg_min;
  max = cfg_max;
  nbins = cfg_nbins;
  ncols = cfg_ncols;
  value = cfg_value;  // Will be updated in make_split
}

template <class T>
__device__ __forceinline__ T SquareFunctor::exec(T x) {
  return MLCommon::myPow(x, (T)2);
}

template <class T>
__device__ __forceinline__ T AbsFunctor::exec(T x) {
  return MLCommon::myAbs(x);
}

float GiniFunctor::exec(std::vector<int> &hist, int nrows) {
  float gval = 1.0;
  for (int i = 0; i < hist.size(); i++) {
    float prob = ((float)hist[i]) / nrows;
    gval -= prob * prob;
  }
  return gval;
}

float EntropyFunctor::exec(std::vector<int> &hist, int nrows) {
  float eval = 0.0;
  for (int i = 0; i < hist.size(); i++) {
    float prob = ((float)hist[i]) / nrows;
    eval += prob * logf(prob);
  }
  return (-1 * eval);
}

__global__ void gini_kernel(const int *__restrict__ labels, const int nrows,
                            const int nmax, int *histout) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  extern __shared__ unsigned int shmemhist[];
  if (threadIdx.x < nmax) shmemhist[threadIdx.x] = 0;

  __syncthreads();

  if (tid < nrows) {
    int label = labels[tid];
    atomicAdd(&shmemhist[label], 1);
  }

  __syncthreads();

  if (threadIdx.x < nmax)
    atomicAdd(&histout[threadIdx.x], shmemhist[threadIdx.x]);

  return;
}

template <typename T>
__global__ void pred_kernel(const T *__restrict__ labels, const int nrows,
                            T *predout) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ T shmempred;

  if (threadIdx.x == 0) shmempred = 0;

  __syncthreads();

  if (tid < nrows) {
    T label = labels[tid];
    atomicAdd(&shmempred, label);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(predout, shmempred);
  }

  return;
}

template <typename T, typename F>
__global__ void mse_kernel(const T *__restrict__ labels, const int nrows,
                           const T *predout, T *mseout) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ T shmemmse;

  if (threadIdx.x == 0) {
    shmemmse = 0;
  }

  __syncthreads();

  if (tid < nrows) {
    T label = labels[tid] - (predout[0] / nrows);
    atomicAdd(&shmemmse, F::exec(label));
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(mseout, shmemmse);
  }

  return;
}

template <typename T, typename F>
void gini(int *labels_in, const int nrows,
          const std::shared_ptr<TemporaryMemory<T, int>> tempmem,
          MetricInfo<T> &split_info, const int unique_labels) {
  int *dhist = tempmem->d_hist->data();
  int *hhist = tempmem->h_hist->data();

  CUDA_CHECK(
    cudaMemsetAsync(dhist, 0, sizeof(int) * unique_labels, tempmem->stream));
  gini_kernel<<<MLCommon::ceildiv(nrows, 128), 128, sizeof(int) * unique_labels,
                tempmem->stream>>>(labels_in, nrows, unique_labels, dhist);
  CUDA_CHECK(cudaGetLastError());
  MLCommon::updateHost(hhist, dhist, unique_labels, tempmem->stream);
  CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

  split_info.hist.resize(unique_labels, 0);
  for (int i = 0; i < unique_labels; i++) {
    split_info.hist[i] = hhist[i];
  }

  split_info.best_metric = F::exec(split_info.hist, nrows);

  return;
}

template <typename T, typename F>
void mse(T *labels_in, const int nrows,
         const std::shared_ptr<TemporaryMemory<T, T>> tempmem,
         MetricInfo<T> &split_info) {
  T *dpred = tempmem->d_predout->data();
  T *dmse = tempmem->d_mseout->data();
  T *hmse = tempmem->h_mseout->data();
  T *hpred = tempmem->h_predout->data();

  CUDA_CHECK(cudaMemsetAsync(dpred, 0, sizeof(T), tempmem->stream));
  CUDA_CHECK(cudaMemsetAsync(dmse, 0, sizeof(T), tempmem->stream));

  pred_kernel<<<MLCommon::ceildiv(nrows, 128), 128, 0, tempmem->stream>>>(
    labels_in, nrows, dpred);
  CUDA_CHECK(cudaGetLastError());
  mse_kernel<T, F><<<MLCommon::ceildiv(nrows, 128), 128, 0, tempmem->stream>>>(
    labels_in, nrows, dpred, dmse);
  CUDA_CHECK(cudaGetLastError());

  MLCommon::updateHost(hmse, dmse, 1, tempmem->stream);
  MLCommon::updateHost(hpred, dpred, 1, tempmem->stream);
  CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

  split_info.best_metric =
    (float)hmse[0] / (float)nrows;  //Update split metric value
  split_info.predict = hpred[0] / (T)nrows;
  return;
}
