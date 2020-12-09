/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <common/allocatorAdapter.hpp>
#include <common/cumlHandle.hpp>

#include <thrust/for_each.h>

#include "pack.h"

namespace ML {
namespace Dbscan {
namespace CorePoints {
namespace Algo {

/**
 * Calculates the core point mask for the batch.
 */
template <typename Index_ = int>
void launcher(const raft::handle_t &handle, Pack<Index_> data,
              Index_ startVertexId, Index_ batchSize, cudaStream_t stream) {
  auto execution_policy =
    ML::thrust_exec_policy(handle.get_device_allocator(), stream);
  auto counting = thrust::make_counting_iterator<int>(0);
  thrust::for_each(execution_policy->on(stream), counting, counting + batchSize,
                   [=] __device__(int idx) {
                     data.mask[idx + startVertexId] =
                       data.vd[idx] >= data.minPts;
                   });
}

}  // namespace Algo
}  // end namespace CorePoints
}  // end namespace Dbscan
}  // namespace ML
