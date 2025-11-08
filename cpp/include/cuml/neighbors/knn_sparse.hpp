/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/distance_type.hpp>
#include <cuml/neighbors/knn.hpp>

#include <cusparse_v2.h>

namespace raft {
class handle_t;
}

namespace ML {
namespace Sparse {

constexpr int DEFAULT_BATCH_SIZE = 1 << 16;

void brute_force_knn(raft::handle_t& handle,
                     const int* idx_indptr,
                     const int* idx_indices,
                     const float* idx_data,
                     size_t idx_nnz,
                     int n_idx_rows,
                     int n_idx_cols,
                     const int* query_indptr,
                     const int* query_indices,
                     const float* query_data,
                     size_t query_nnz,
                     int n_query_rows,
                     int n_query_cols,
                     int* output_indices,
                     float* output_dists,
                     int k,
                     size_t batch_size_index           = DEFAULT_BATCH_SIZE,
                     size_t batch_size_query           = DEFAULT_BATCH_SIZE,
                     ML::distance::DistanceType metric = ML::distance::DistanceType::L2Expanded,
                     float metricArg                   = 0);
};  // end namespace Sparse
};  // end namespace ML
