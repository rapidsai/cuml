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
#define LEAF 0xFFFFFFFF
#define PUSHRIGHT 0x00000001

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

template <typename T>
__global__ void split_level_kernel(
  const T* __restrict__ data, const T* __restrict__ quantile,
  const int* __restrict__ split_col_index,
  const int* __restrict__ split_bin_index, const int nrows, const int ncols,
  const int nbins, const int n_nodes,
  const unsigned int* __restrict__ new_node_flags,
  unsigned int* __restrict__ flags) {
  unsigned int threadid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int local_flag;

  for (int tid = threadid; tid < nrows; tid += gridDim.x * blockDim.x) {
    if (tid < nrows) {
      local_flag = flags[tid];
    } else {
      local_flag = LEAF;
    }

    if (local_flag != LEAF) {
      unsigned int local_leaf_flag = new_node_flags[local_flag];
      if (local_leaf_flag != LEAF) {
        int colidx = split_col_index[local_flag];
        T quesval = quantile[colidx * nbins + split_bin_index[local_flag]];
        T local_data = data[colidx * nrows + tid];
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
