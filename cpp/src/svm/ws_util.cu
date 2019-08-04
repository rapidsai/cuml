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

#include "ml_utils.h"

#include <cuda_utils.h>
#include <limits.h>
#include <cub/cub.cuh>

namespace ML {
namespace SVM {

__global__ void range(int *f_idx, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    f_idx[tid] = tid;
  }
}

__global__ void map_to_sorted(const bool *available, int n_rows,
                              bool *available_sorted, const int *idx_sorted) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_rows) {
    int idx = idx_sorted[tid];
    available_sorted[tid] = available[idx];
  }
}
__global__ void set_unavailable(bool *available, int n_rows, const int *idx,
                                int n_selected) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_selected) {
    available[idx[tid]] = false;
  }
}

__global__ void update_priority(int *new_priority, int n_selected,
                                const int *new_idx, int n_ws, const int *idx,
                                const int *priority) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_selected) {
    int my_new_idx = new_idx[tid];
    // The working set size is limited (~1024 elements) so we just loop through it
    for (int i = 0; i < n_ws; i++) {
      if (idx[i] == my_new_idx) new_priority[tid] = priority[i] + 1;
    }
  }
}
}  // namespace SVM
}  // namespace ML
