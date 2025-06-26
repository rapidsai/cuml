/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuml/common/distance_type.hpp>
#include <cuml/neighbors/knn_sparse.hpp>

#include <raft/core/handle.hpp>

#include <cuvs/neighbors/brute_force.hpp>

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
                     ML::distance::DistanceType metric,
                     float metricArg)
{
  auto idx_structure = raft::make_device_compressed_structure_view<int, int, int>(
    const_cast<int*>(idx_indptr), const_cast<int*>(idx_indices), n_idx_rows, n_idx_cols, idx_nnz);
  auto idx_csr = raft::make_device_csr_matrix_view<const float>(idx_data, idx_structure);

  auto query_structure =
    raft::make_device_compressed_structure_view<int, int, int>(const_cast<int*>(query_indptr),
                                                               const_cast<int*>(query_indices),
                                                               n_query_rows,
                                                               n_query_cols,
                                                               query_nnz);
  auto query_csr = raft::make_device_csr_matrix_view<const float>(query_data, query_structure);

  cuvs::neighbors::brute_force::sparse_search_params search_params;
  search_params.batch_size_index = batch_size_index;
  search_params.batch_size_query = batch_size_query;

  auto index = cuvs::neighbors::brute_force::build(
    handle, idx_csr, static_cast<cuvs::distance::DistanceType>(metric), metricArg);

  cuvs::neighbors::brute_force::search(
    handle,
    search_params,
    index,
    query_csr,
    raft::make_device_matrix_view<int, int64_t>(output_indices, n_query_rows, k),
    raft::make_device_matrix_view<float, int64_t>(output_dists, n_query_cols, k));
}
};  // namespace Sparse
};  // namespace ML
