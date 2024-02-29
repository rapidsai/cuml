/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cuml/tsa/holtwinters_params.h>

#include <raft/linalg/eltwise.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#define IDX(n, m, N) (n + (m) * (N))

#define STMP_EPS (1e-6)

#define GOLD \
  0.381966011250105151795413165634361882279690820194237137864551377294739537181097550292792795810608862515245
#define PG_EPS 1e-10

#define SUBSTITUTE(a, b, c, d) \
  (a) = (b);                   \
  (c) = (d);

#define MAX_BLOCKS_PER_DIM 65535

#define GET_TID (blockIdx.x * blockDim.x + threadIdx.x)

inline int GET_THREADS_PER_BLOCK(const int n, const int max_threads = 512)
{
  int ret;
  if (n <= 128)
    ret = 32;
  else if (n <= 1024)
    ret = 128;
  else
    ret = 512;
  return ret > max_threads ? max_threads : ret;
}

inline int GET_NUM_BLOCKS(const int n,
                          const int max_threads = 512,
                          const int max_blocks  = MAX_BLOCKS_PER_DIM)
{
  int ret = (n - 1) / GET_THREADS_PER_BLOCK(n, max_threads) + 1;
  return ret > max_blocks ? max_blocks : ret;
}

template <typename Dtype>
__device__ Dtype abs_device(Dtype val)
{
  int nbytes = sizeof(val);
  if (nbytes == sizeof(float))
    return fabsf(val);
  else
    return fabs(val);
}

template <typename Dtype>
__device__ Dtype bound_device(Dtype val, Dtype min = .0, Dtype max = 1.)
{
  int nbytes = sizeof(val);
  if (nbytes == sizeof(float))
    return fminf(fmaxf(val, min), max);
  else
    return fmin(fmax(val, min), max);
}

template <typename Dtype>
__device__ Dtype max3(Dtype a, Dtype b, Dtype c)
{
  return a > b ? (a > c ? a : c) : (b > c ? b : c);
}
