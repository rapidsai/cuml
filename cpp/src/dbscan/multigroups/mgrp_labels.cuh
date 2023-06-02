/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace Labels {

template <typename index_t = int, size_t TPB = 128>
__global__ void label_bias_kernel(const index_t* labels,
                                  index_t* lbl_bias,
                                  const index_t* lbl_start_ids,
                                  index_t n_groups,
                                  index_t MAX_LABEL)
{
  typedef cub::BlockReduce<index_t, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  index_t group_id = blockIdx.x;
  index_t tid      = threadIdx.x;
  if (group_id > n_groups) return;

  index_t min_value  = MAX_LABEL;
  index_t valid_step = lbl_start_ids[group_id + 1] - lbl_start_ids[group_id];
  for (index_t tid = threadIdx.x; tid < (valid_step + TPB - 1) / TPB * TPB; tid += TPB) {
    index_t thread_data = (tid < valid_step) ? labels[lbl_start_ids[group_id] + tid] : MAX_LABEL;
    index_t aggregate   = BlockReduce(temp_storage).Reduce(thread_data, cub::Min());
    min_value           = (min_value < aggregate) ? min_value : aggregate;
    __syncthreads();
  }

  if (threadIdx.x == 0) { lbl_bias[group_id] = min_value; }
}

template <typename index_t = int>
__global__ void multiGroupRelabelForSklKernel(index_t* labels,
                                              const index_t* lbl_start_ids,
                                              const index_t* lbl_valid_steps,
                                              const index_t* lbl_bias,
                                              index_t n_groups,
                                              index_t n_points,
                                              index_t MAX_LABEL)
{
  index_t group_id = blockIdx.y * blockDim.y + threadIdx.y;
  index_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_id >= n_groups) return;

  index_t valid_step = lbl_valid_steps[group_id];
  index_t bias       = lbl_bias[group_id];
  index_t label_id   = lbl_start_ids[group_id] + tid;
  if (tid < valid_step && label_id < n_points) {
    if (labels[label_id] == MAX_LABEL) {
      labels[label_id] = -1;
    } else {
      labels[label_id] -= bias;
    }
  }
}

}  // namespace Labels
}  // namespace Multigroups
}  // namespace Dbscan
}  // namespace ML