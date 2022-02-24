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

#pragma once

#include <limits>
#include <math.h>

#include <linalg/init.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/interruptible.hpp>

namespace MLCommon {
namespace Label {

/** Note: this is one possible implementation where we represent the label
 *  equivalence graph implicitly using labels_a, labels_b and mask.
 *  For an additional cost we can build the graph with edges
 *  E={(A[i], B[i]) | M[i]=1} and make this step faster */
template <typename Index_, int TPB_X = 256>
__global__ void __launch_bounds__(TPB_X) propagate_label_kernel(const Index_* __restrict__ labels_a,
                                                                const Index_* __restrict__ labels_b,
                                                                Index_* __restrict__ R,
                                                                const bool* __restrict__ mask,
                                                                bool* __restrict__ m,
                                                                Index_ N)
{
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
__global__ void __launch_bounds__(TPB_X) reassign_label_kernel(Index_* __restrict__ labels_a,
                                                               const Index_* __restrict__ labels_b,
                                                               const Index_* __restrict__ R,
                                                               Index_ N,
                                                               Index_ MAX_LABEL)
{
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    // Note: labels are from 1 to N
    Index_ la     = labels_a[tid];
    Index_ lb     = __ldg(labels_b + tid);
    Index_ ra     = (la == MAX_LABEL) ? MAX_LABEL : __ldg(R + (la - 1)) + 1;
    Index_ rb     = (lb == MAX_LABEL) ? MAX_LABEL : __ldg(R + (lb - 1)) + 1;
    labels_a[tid] = min(ra, rb);
  }
}

/**
 * @brief Merge two labellings in-place, according to a core mask
 *
 * A labelling is a representation of disjoint sets (groups) where points that
 * belong to the same group have the same label. It is assumed that group
 * labels take values between 1 and N. Labels relate to points, i.e a label i+1
 * means that you belong to the same group as the point i.
 * The special value MAX_LABEL is used to mark points that are not labelled.
 *
 * The two label arrays A and B induce two sets of groups over points 0..N-1.
 * If a point is labelled i in A and j in B and the mask is true for this
 * point, then i and j are equivalent labels and their groups are merged by
 * relabeling the elements of both groups to have the same label. The new label
 * is the smaller one from the original labels.
 * It is required that if the mask is true for a point, this point is labelled
 * (i.e its label is different than the special value MAX_LABEL).
 *
 * One use case is finding connected components: the two input label arrays can
 * represent the connected components of graphs G_A and G_B, and the output
 * would be the connected components labels of G_A \union G_B.
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
void merge_labels(Index_* labels_a,
                  const Index_* labels_b,
                  const bool* mask,
                  Index_* R,
                  bool* m,
                  Index_ N,
                  cudaStream_t stream)
{
  dim3 blocks(raft::ceildiv(N, Index_(TPB_X)));
  dim3 threads(TPB_X);
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  // Initialize R. R defines the relabeling rules; after merging the input
  // arrays, label l will be reassigned as R[l-1]+1.
  MLCommon::LinAlg::range(R, N, stream);

  // We define the label equivalence graph: G = (V, E), where:
  //  - V is the set of unique values from labels_a and labels_b
  //  - E = {(labels_a[k], labels_b[k]) | mask[k] == true and k \in 0..n-1 }
  // The edges connect groups from the two labellings. Only points with true
  // mask can induce connection between groups.

  // Step 1: compute connected components in the label equivalence graph
  bool host_m;
  do {
    RAFT_CUDA_TRY(cudaMemsetAsync(m, false, sizeof(bool), stream));

    propagate_label_kernel<Index_, TPB_X>
      <<<blocks, threads, 0, stream>>>(labels_a, labels_b, R, mask, m, N);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    raft::update_host(&host_m, m, 1, stream);
    raft::interruptible::synchronize(stream);
  } while (host_m);

  // Step 2: re-assign minimum equivalent label
  reassign_label_kernel<Index_, TPB_X>
    <<<blocks, threads, 0, stream>>>(labels_a, labels_b, R, N, MAX_LABEL);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

};  // namespace Label
};  // namespace MLCommon
