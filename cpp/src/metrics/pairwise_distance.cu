
/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cuml/metrics/metrics.hpp>

#include <raft/handle.hpp>

#include <raft/distance/distance.hpp>
#include <raft/sparse/distance/common.h>
#include <raft/sparse/distance/distance.cuh>

namespace ML {

namespace Metrics {
void pairwise_distance(const raft::handle_t& handle,
                       const double* x,
                       const double* y,
                       double* dist,
                       int m,
                       int n,
                       int k,
                       raft::distance::DistanceType metric,
                       bool isRowMajor,
                       double metric_arg)
{
    raft::distance::pairwise_distance<double, int>(
      handle, x, y, dist, m, n, k, raft::distance::DistanceType::Canberra, isRowMajor);
}

void pairwise_distance(const raft::handle_t& handle,
                       const float* x,
                       const float* y,
                       float* dist,
                       int m,
                       int n,
                       int k,
                       raft::distance::DistanceType metric,
                       bool isRowMajor,
                       float metric_arg)
{
  raft::distance::pairwise_distance<float, int>(
    handle, x, y, dist, m, n, k, raft::distance::DistanceType::Canberra, isRowMajor);
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
                             raft::distance::DistanceType metric,
                             float metric_arg)
{
  raft::sparse::distance::distances_config_t<value_idx, value_t> dist_config(handle);

  dist_config.b_nrows   = x_nrows;
  dist_config.b_ncols   = n_cols;
  dist_config.b_nnz     = x_nnz;
  dist_config.b_indptr  = x_indptr;
  dist_config.b_indices = x_indices;
  dist_config.b_data    = x;

  dist_config.a_nrows   = y_nrows;
  dist_config.a_ncols   = n_cols;
  dist_config.a_nnz     = y_nnz;
  dist_config.a_indptr  = y_indptr;
  dist_config.a_indices = y_indices;
  dist_config.a_data    = y;

  raft::sparse::distance::pairwiseDistance(dist, dist_config, metric, metric_arg);
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
                             raft::distance::DistanceType metric,
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
                             raft::distance::DistanceType metric,
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
