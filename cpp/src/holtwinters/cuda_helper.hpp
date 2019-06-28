/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#define MAX_BLOCKS_PER_DIM 65535
#define GPU_LOOP(i, n) \
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < (n); i += blockDim.x*gridDim.x)
#define GET_TID (blockIdx.x*blockDim.x+threadIdx.x)

inline int GET_THREADS_PER_BLOCK(const int n, const int max_threads = 512) {
  int ret;
  if (n <= 128)
    ret = 32;
  else if (n <= 1024)
    ret = 128;
  else
    ret = 512;
  return ret > max_threads ? max_threads : ret;
}

inline int GET_NUM_BLOCKS(const int n, const int max_threads = 512, const int max_blocks = MAX_BLOCKS_PER_DIM) {
  int ret = (n-1)/GET_THREADS_PER_BLOCK(n, max_threads)+1;
  return ret > max_blocks ? max_blocks : ret;
}
