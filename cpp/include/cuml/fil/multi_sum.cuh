/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
/** @file multi_sum.cuh */
#pragma once
#include <raft/cuda_utils.cuh>
/**
 template parameters: data [T]ype, reduction [R]adix
 function parameters:
 @data[] holds one value per thread in shared memory
 @n_groups is the number of indendent reductions
 @n_values is the size of each individual reduction,
   that is the number of values to be reduced to a single value
 function returns: one sum per thread, for @n_groups first threads.
 important: @data[] is "spoiled" during the process: at the end,
   it will contain neither the initial nor the final values. the only valid
   result is the one returned by the function. That makes it faster.
 other assumptions:
 data[n_groups * n_values - 1] is within range
 T::operator+= is defined, and the implied addition is associative.
 @data[] layout assumption:
 @data[] values are ordered such that the stride is 1 for values belonging
   to the same group and @n_groups for values that are to be added together
*/
template <int R = 5, typename T>
__device__ T multi_sum(T* data, int n_groups, int n_values)
{
  T acc = threadIdx.x < n_groups * n_values ? data[threadIdx.x] : T();
  while (n_values > 1) {
    // n_targets is the number of values per group after the end of this iteration
    int n_targets = raft::ceildiv(n_values, R);
    if (threadIdx.x < n_targets * n_groups) {
#pragma unroll
      for (int i = 1; i < R; ++i) {
        int idx = threadIdx.x + i * n_targets * n_groups;
        if (idx < n_values * n_groups) acc += data[idx];
      }
      if (n_targets > 1) data[threadIdx.x] = acc;
    }
    n_values = n_targets;
    if (n_values > 1) __syncthreads();
  }
  return acc;
}
