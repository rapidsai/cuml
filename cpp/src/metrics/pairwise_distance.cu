
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
#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/distance/distance.cuh>

#include <cuvs/distance/distance.hpp>

namespace ML {

namespace Metrics {
void pairwise_distance(const raft::handle_t& handle,
                       const double* x,
                       const double* y,
                       double* dist,
                       int m,
                       int n,
                       int k,
                       ML::distance::DistanceType metric,
                       bool isRowMajor,
                       double metric_arg)
{
  if (isRowMajor) {
    cuvs::distance::pairwise_distance(
      handle,
      raft::make_device_matrix_view<const double, int64_t, raft::row_major>(x, m, k),
      raft::make_device_matrix_view<const double, int64_t, raft::row_major>(y, n, k),
      raft::make_device_matrix_view<double, int64_t, raft::row_major>(dist, m, n),
      static_cast<cuvs::distance::DistanceType>(metric),
      metric_arg);
  } else {
    cuvs::distance::pairwise_distance(
      handle,
      raft::make_device_matrix_view<const double, int64_t, raft::col_major>(x, m, k),
      raft::make_device_matrix_view<const double, int64_t, raft::col_major>(y, n, k),
      raft::make_device_matrix_view<double, int64_t, raft::col_major>(dist, m, n),
      static_cast<cuvs::distance::DistanceType>(metric),
      metric_arg);
  }
}

void pairwise_distance(const raft::handle_t& handle,
                       const float* x,
                       const float* y,
                       float* dist,
                       int m,
                       int n,
                       int k,
                       ML::distance::DistanceType metric,
                       bool isRowMajor,
                       float metric_arg)
{
  if (isRowMajor) {
    cuvs::distance::pairwise_distance(
      handle,
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(x, m, k),
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(y, n, k),
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(dist, m, n),
      static_cast<cuvs::distance::DistanceType>(metric),
      metric_arg);
  } else {
    cuvs::distance::pairwise_distance(
      handle,
      raft::make_device_matrix_view<const float, int64_t, raft::col_major>(x, m, k),
      raft::make_device_matrix_view<const float, int64_t, raft::col_major>(y, n, k),
      raft::make_device_matrix_view<float, int64_t, raft::col_major>(dist, m, n),
      static_cast<cuvs::distance::DistanceType>(metric),
      metric_arg);
  }
}

template <typename value_idx = int, typename value_t = float>
void pairwiseDistance_sparse(const raft::handle_t& handle,
                             value_t* x,
                             value_t* y,
                             value_t* dist,
                             value_idx x_nrows,
                             value_idx y_nrows,
                             value_idx n_cols,
                             value_idx x_nnz,
                             value_idx y_nnz,
                             value_idx* x_indptr,
                             value_idx* y_indptr,
                             value_idx* x_indices,
                             value_idx* y_indices,
                             ML::distance::DistanceType metric,
                             float metric_arg)
{
  auto out = raft::make_device_matrix_view<value_t, value_idx>(dist, y_nrows, x_nrows);

  auto x_structure = raft::make_device_compressed_structure_view<value_idx, value_idx, value_idx>(
    x_indptr, x_indices, x_nrows, n_cols, x_nnz);
  auto x_csr_view = raft::make_device_csr_matrix_view<const value_t>(x, x_structure);

  auto y_structure = raft::make_device_compressed_structure_view<value_idx, value_idx, value_idx>(
    y_indptr, y_indices, y_nrows, n_cols, y_nnz);
  auto y_csr_view = raft::make_device_csr_matrix_view<const value_t>(y, y_structure);

  cuvs::distance::pairwise_distance(handle,
                                    y_csr_view,
                                    x_csr_view,
                                    out,
                                    static_cast<cuvs::distance::DistanceType>(metric),
                                    metric_arg);
}

void pairwiseDistance_sparse(const raft::handle_t& handle,
                             float* x,
                             float* y,
                             float* dist,
                             int x_nrows,
                             int y_nrows,
                             int n_cols,
                             int x_nnz,
                             int y_nnz,
                             int* x_indptr,
                             int* y_indptr,
                             int* x_indices,
                             int* y_indices,
                             ML::distance::DistanceType metric,
                             float metric_arg)
{
  pairwiseDistance_sparse<int, float>(handle,
                                      x,
                                      y,
                                      dist,
                                      x_nrows,
                                      y_nrows,
                                      n_cols,
                                      x_nnz,
                                      y_nnz,
                                      x_indptr,
                                      y_indptr,
                                      x_indices,
                                      y_indices,
                                      metric,
                                      metric_arg);
}

void pairwiseDistance_sparse(const raft::handle_t& handle,
                             double* x,
                             double* y,
                             double* dist,
                             int x_nrows,
                             int y_nrows,
                             int n_cols,
                             int x_nnz,
                             int y_nnz,
                             int* x_indptr,
                             int* y_indptr,
                             int* x_indices,
                             int* y_indices,
                             ML::distance::DistanceType metric,
                             float metric_arg)
{
  pairwiseDistance_sparse<int, double>(handle,
                                       x,
                                       y,
                                       dist,
                                       x_nrows,
                                       y_nrows,
                                       n_cols,
                                       x_nnz,
                                       y_nnz,
                                       x_indptr,
                                       y_indptr,
                                       x_indices,
                                       y_indices,
                                       metric,
                                       metric_arg);
}
}  // namespace Metrics
}  // namespace ML
