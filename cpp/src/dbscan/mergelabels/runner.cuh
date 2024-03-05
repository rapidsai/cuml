/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/label/merge_labels.cuh>
namespace ML {
namespace Dbscan {
namespace MergeLabels {

/**
 * Merges to label arrays according to a given core point mask
 * @param[in]    handle      raft handle
 * @param[inout] labels_a    First input, and output label array (in-place)
 * @param[in]    labels_b    Second input label array
 * @param[in]    mask        Core point mask
 * @param[in]    work_buffer Working buffer (for R)
 * @param[in]    m           Working flag
 * @param[in]    N           Number of points in the dataset
 * @param[in]    stream      CUDA stream
 */
template <typename Index_ = int, int TPB_X = 256>
void run(const raft::handle_t& handle,
         Index_* labels_a,
         const Index_* labels_b,
         const bool* mask,
         Index_* work_buffer,
         bool* m,
         Index_ N,
         cudaStream_t stream)
{
  raft::label::merge_labels<Index_, TPB_X>(labels_a, labels_b, mask, work_buffer, m, N, stream);
}

}  // namespace MergeLabels
}  // namespace Dbscan
}  // namespace ML
