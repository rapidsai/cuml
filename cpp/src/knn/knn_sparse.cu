/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include <cuml/neighbors/knn_sparse.hpp>

#include <raft/sparse/selection/knn.cuh>
#include <raft/spatial/knn/specializations.hpp>

#include <cusparse_v2.h>

namespace ML {
namespace Sparse {

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
                     size_t batch_size_index,  // approx 1M
                     size_t batch_size_query,
                     raft::distance::DistanceType metric,
                     float metricArg)
{
  raft::sparse::selection::brute_force_knn(idx_indptr,
                                           idx_indices,
                                           idx_data,
                                           idx_nnz,
                                           n_idx_rows,
                                           n_idx_cols,
                                           query_indptr,
                                           query_indices,
                                           query_data,
                                           query_nnz,
                                           n_query_rows,
                                           n_query_cols,
                                           output_indices,
                                           output_dists,
                                           k,
                                           handle,
                                           batch_size_index,
                                           batch_size_query,
                                           metric,
                                           metricArg);
}
};  // namespace Sparse
};  // namespace ML
