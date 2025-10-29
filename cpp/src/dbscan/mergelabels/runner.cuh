/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
