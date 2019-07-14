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

  for (int tid = threadid; tid < nrows; tid += blockDim.x * gridDim.x) {
    T label = labels[tid];
    unsigned int count = sample_cnt[tid];
    T value = F::exec(label - (predout[0] / count[0]));
    atomicAdd(&shmemmse, count * value);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(mseout, shmemmse);
  }
  return;
}
