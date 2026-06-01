/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>

namespace ML {

/**
 * @brief Compute per-tree sample weights for class_weight='balanced_subsample'.
 *
 * Mirrors sklearn's _parallel_build_trees: bincount the bootstrap labels,
 * derive balanced reciprocals using `n_classes_present` (classes with
 * count > 0 in the bootstrap, matching sklearn's len(classes_subsample)),
 * then fill per-row weights by multiplying the per-class reciprocal into
 * `base_sample_weight`.
 *
 * @note `class_weight_array` is accepted on the ABI but not consumed by
 * this mode (forward-compat for future C++ modes).
 *
 * @param[in]  handle             RAFT handle (for memory resource).
 * @param[in]  stream             Stream the OMP loop is using for this tree.
 * @param[in]  labels             [n_rows] device pointer; integer class labels.
 * @param[in]  selected_rows      [n_sampled_rows] device pointer; bootstrap indices.
 * @param[in]  n_sampled_rows     length of selected_rows.
 * @param[in]  n_unique_labels    number of classes (upper bound of label value + 1).
 * @param[in]  n_rows             length of labels / per_tree_weights.
 * @param[in]  base_sample_weight [n_rows] device pointer, or nullptr (treated as 1.0).
 * @param[in]  class_weight_array [n_unique_labels] device pointer (currently unused; see above).
 * @param[out] per_tree_weights   [n_rows] device pointer, pre-allocated; receives the result.
 */
template <typename DataT, typename LabelT>
void computePerTreeBalancedWeights(const raft::handle_t& handle,
                                   cudaStream_t stream,
                                   const LabelT* labels,
                                   const int* selected_rows,
                                   int n_sampled_rows,
                                   int n_unique_labels,
                                   int n_rows,
                                   const DataT* base_sample_weight,
                                   const double* class_weight_array,
                                   DataT* per_tree_weights);

}  // namespace ML
