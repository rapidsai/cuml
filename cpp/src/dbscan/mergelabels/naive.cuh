/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <math.h>
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>

#include "pack.h"

namespace ML {
namespace Dbscan {
namespace MergeLabels {
namespace Naive {

/** Note: this is one possible implementation. For an additional cost we can
 *  build the graph with edges E={(A[i], B[i]) | M[i]=1} and make this step
 *  faster */
template <typename Index_, int TPB_X = 32>
__global__ void __launch_bounds__(TPB_X)
  propagate_label_kernel(const Index_* __restrict__ labelsA,
                         const Index_* __restrict__ labelsB,
                         Index_* __restrict__ R, const bool* __restrict__ mask,
                         bool* __restrict__ m, Index_ N) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    if (__ldg((char*)mask + tid)) {
      // Note: labels are from 1 to N
      Index_ la = __ldg(labelsA + tid) - 1;
      Index_ lb = __ldg(labelsB + tid) - 1;
      Index_ ra = R[la];
      Index_ rb = R[lb];
      if (ra != rb) {
        *m = true;
        // min(ra, rb) would be sufficient but this speeds up convergence
        Index_ rmin = R[min(ra, rb)];
        if (sizeof(Index_) == 4) {
          atomicMin((int*)(R + ra), rmin);
          atomicMin((int*)(R + rb), rmin);
        } else if (sizeof(Index_) == 8) {
          atomicMin((long long int*)(R + ra), rmin);
          atomicMin((long long int*)(R + rb), rmin);
        }
      }
    }
  }
}

template <typename Index_, int TPB_X = 32>
__global__ void __launch_bounds__(TPB_X)
  reassign_label_kernel(Index_* __restrict__ labelsA,
                        const Index_* __restrict__ labelsB,
                        const Index_* __restrict__ R, Index_ N,
                        Index_ MAX_LABEL) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    // Note: labels are from 1 to N
    Index_ la = labelsA[tid];
    Index_ lb = __ldg(labelsB + tid);
    Index_ ra = (la == MAX_LABEL) ? MAX_LABEL : __ldg(R + (la - 1)) + 1;
    Index_ rb = (lb == MAX_LABEL) ? MAX_LABEL : __ldg(R + (lb - 1)) + 1;
    labelsA[tid] = min(ra, rb);
  }
}

/**
 * TODO: docs
 */
template <typename Index_ = int, int TPB_X = 32>
void launcher(const raft::handle_t& handle, Pack<Index_> data,
              cudaStream_t stream) {
  auto execution_policy =
    ML::thrust_exec_policy(handle.get_device_allocator(), stream);
  dim3 blocks(raft::ceildiv(data.N, Index_(TPB_X)));
  dim3 threads(TPB_X);
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  Index_* R = data.workBuffer;

  // Initialize R
  auto counting = thrust::make_counting_iterator<int>(0);
  thrust::for_each(execution_policy->on(stream), counting, counting + data.N,
                   [=] __device__(int idx) { R[idx] = idx; });

  // Connected components in the label equivalence graph
  bool host_m;
  do {
    CUDA_CHECK(cudaMemsetAsync(data.m, false, sizeof(bool), stream));

    propagate_label_kernel<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
      data.labelsA, data.labelsB, R, data.mask, data.m, data.N);
    CUDA_CHECK(cudaPeekAtLastError());

    raft::update_host(&host_m, data.m, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } while (host_m);

  // Re-assign minimum equivalent label
  reassign_label_kernel<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
    data.labelsA, data.labelsB, R, data.N, MAX_LABEL);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Naive
}  // namespace MergeLabels
}  // namespace Dbscan
}  // namespace ML
