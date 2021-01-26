/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <math.h>
#include <limits>

#include <linalg/init.h>
#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

namespace MLCommon {
namespace Label {

/** Note: this is one possible implementation. For an additional cost we can
 *  build the graph with edges E={(A[i], B[i]) | M[i]=1} and make this step
 *  faster */
template <typename Index_, int TPB_X = 256>
__global__ void __launch_bounds__(TPB_X)
  propagate_label_kernel(const Index_* __restrict__ labels_a,
                         const Index_* __restrict__ labels_b,
                         Index_* __restrict__ R, const bool* __restrict__ mask,
                         bool* __restrict__ m, Index_ N) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    if (__ldg((char*)mask + tid)) {
      // Note: labels are from 1 to N
      Index_ la = __ldg(labels_a + tid) - 1;
      Index_ lb = __ldg(labels_b + tid) - 1;
      Index_ ra = R[la];
      Index_ rb = R[lb];
      if (ra != rb) {
        *m = true;
        // min(ra, rb) would be sufficient but this speeds up convergence
        Index_ rmin = R[min(ra, rb)];
        if (sizeof(Index_) == 4) {
          atomicMin((int*)(R + la), rmin);
          atomicMin((int*)(R + lb), rmin);
        } else if (sizeof(Index_) == 8) {
          atomicMin((long long int*)(R + la), rmin);
          atomicMin((long long int*)(R + lb), rmin);
        }
      }
    }
  }
}

template <typename Index_, int TPB_X = 256>
__global__ void __launch_bounds__(TPB_X)
  reassign_label_kernel(Index_* __restrict__ labels_a,
                        const Index_* __restrict__ labels_b,
                        const Index_* __restrict__ R, Index_ N,
                        Index_ MAX_LABEL) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    // Note: labels are from 1 to N
    Index_ la = labels_a[tid];
    Index_ lb = __ldg(labels_b + tid);
    Index_ ra = (la == MAX_LABEL) ? MAX_LABEL : __ldg(R + (la - 1)) + 1;
    Index_ rb = (lb == MAX_LABEL) ? MAX_LABEL : __ldg(R + (lb - 1)) + 1;
    labels_a[tid] = min(ra, rb);
  }
}

/**
 * Merge two label arrays in-place, according to a core mask
 *
 * The input arrays describe connected components. If a core point
 * is labelled i in A and j in B, i and j are equivalent labels and their
 * connected components are merged
 *
 * @param[inout] labels_a    First input, and output label array (in-place)
 * @param[in]    labels_b    Second input label array
 * @param[in]    mask        Core point mask
 * @param[out]   R           Label equivalence map
 * @param[in]    m           Working flag
 * @param[in]    N           Number of points in the dataset
 * @param[in]    stream      CUDA stream
 */
template <typename Index_ = int, int TPB_X = 256>
void merge_labels(Index_* labels_a, const Index_* labels_b, const bool* mask,
                  Index_* R, bool* m, Index_ N, cudaStream_t stream) {
  dim3 blocks(raft::ceildiv(N, Index_(TPB_X)));
  dim3 threads(TPB_X);
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  // Initialize R
  MLCommon::LinAlg::range(R, N, stream);

  // Connected components in the label equivalence graph
  bool host_m;
  do {
    CUDA_CHECK(cudaMemsetAsync(m, false, sizeof(bool), stream));

    propagate_label_kernel<Index_, TPB_X>
      <<<blocks, threads, 0, stream>>>(labels_a, labels_b, R, mask, m, N);
    CUDA_CHECK(cudaPeekAtLastError());

    raft::update_host(&host_m, m, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } while (host_m);

  // Re-assign minimum equivalent label
  reassign_label_kernel<Index_, TPB_X>
    <<<blocks, threads, 0, stream>>>(labels_a, labels_b, R, N, MAX_LABEL);
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // namespace Label
};  // end namespace MLCommon