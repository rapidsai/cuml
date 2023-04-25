/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <cuml/cluster/dbscan.hpp>

#include "dbscan.cuh"
#include <raft/util/cudart_utils.hpp>

namespace ML {
namespace Dbscan {

void fit(const raft::handle_t& handle,
         float* input,
         int n_rows,
         int n_cols,
         float eps,
         int min_pts,
         raft::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices,
         size_t max_bytes_per_batch,
         int verbosity,
         bool opg)
{
  if (opg)
    dbscanFitImpl<float, int, true>(handle,
                                    input,
                                    n_rows,
                                    n_cols,
                                    eps,
                                    min_pts,
                                    metric,
                                    labels,
                                    core_sample_indices,
                                    max_bytes_per_batch,
                                    handle.get_stream(),
                                    verbosity);
  else
    dbscanFitImpl<float, int, false>(handle,
                                     input,
                                     n_rows,
                                     n_cols,
                                     eps,
                                     min_pts,
                                     metric,
                                     labels,
                                     core_sample_indices,
                                     max_bytes_per_batch,
                                     handle.get_stream(),
                                     verbosity);
}

void fit(const raft::handle_t& handle,
         double* input,
         int n_rows,
         int n_cols,
         double eps,
         int min_pts,
         raft::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices,
         size_t max_bytes_per_batch,
         int verbosity,
         bool opg)
{
  if (opg)
    dbscanFitImpl<double, int, true>(handle,
                                     input,
                                     n_rows,
                                     n_cols,
                                     eps,
                                     min_pts,
                                     metric,
                                     labels,
                                     core_sample_indices,
                                     max_bytes_per_batch,
                                     handle.get_stream(),
                                     verbosity);
  else
    dbscanFitImpl<double, int, false>(handle,
                                      input,
                                      n_rows,
                                      n_cols,
                                      eps,
                                      min_pts,
                                      metric,
                                      labels,
                                      core_sample_indices,
                                      max_bytes_per_batch,
                                      handle.get_stream(),
                                      verbosity);
}

void fit(const raft::handle_t& handle,
         float* input,
         int64_t n_rows,
         int64_t n_cols,
         float eps,
         int min_pts,
         raft::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices,
         size_t max_bytes_per_batch,
         int verbosity,
         bool opg)
{
  if (opg)
    dbscanFitImpl<float, int64_t, true>(handle,
                                        input,
                                        n_rows,
                                        n_cols,
                                        eps,
                                        min_pts,
                                        metric,
                                        labels,
                                        core_sample_indices,
                                        max_bytes_per_batch,
                                        handle.get_stream(),
                                        verbosity);
  else
    dbscanFitImpl<float, int64_t, false>(handle,
                                         input,
                                         n_rows,
                                         n_cols,
                                         eps,
                                         min_pts,
                                         metric,
                                         labels,
                                         core_sample_indices,
                                         max_bytes_per_batch,
                                         handle.get_stream(),
                                         verbosity);
}

void fit(const raft::handle_t& handle,
         double* input,
         int64_t n_rows,
         int64_t n_cols,
         double eps,
         int min_pts,
         raft::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices,
         size_t max_bytes_per_batch,
         int verbosity,
         bool opg)
{
  if (opg)
    dbscanFitImpl<double, int64_t, true>(handle,
                                         input,
                                         n_rows,
                                         n_cols,
                                         eps,
                                         min_pts,
                                         metric,
                                         labels,
                                         core_sample_indices,
                                         max_bytes_per_batch,
                                         handle.get_stream(),
                                         verbosity);
  else
    dbscanFitImpl<double, int64_t, false>(handle,
                                          input,
                                          n_rows,
                                          n_cols,
                                          eps,
                                          min_pts,
                                          metric,
                                          labels,
                                          core_sample_indices,
                                          max_bytes_per_batch,
                                          handle.get_stream(),
                                          verbosity);
}

void fit(const raft::handle_t& handle,
         float* input,
         int n_groups,
         int* n_rows_ptr,
         int n_cols,
         const float* eps_ptr,
         const int* min_pts_ptr,
         raft::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices,
         size_t max_bytes_per_batch,
         int verbosity,
         void* custom_workspace,
         size_t* custom_workspace_size,
         bool opg)
{
  ASSERT(!opg, "DBSCAN for multi-groups doesn't support multi-GPU");
  dbscanFitImpl<float, int, false>(handle,
                                   input,
                                   n_groups,
                                   n_rows_ptr,
                                   n_cols,
                                   eps_ptr,
                                   min_pts_ptr,
                                   metric,
                                   labels,
                                   core_sample_indices,
                                   max_bytes_per_batch / 1e6,
                                   handle.get_stream(),
                                   verbosity,
                                   custom_workspace,
                                   custom_workspace_size);
}

void fit(const raft::handle_t& handle,
         double* input,
         int n_groups,
         int* n_rows_ptr,
         int n_cols,
         const double* eps_ptr,
         const int* min_pts_ptr,
         raft::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices,
         size_t max_bytes_per_batch,
         int verbosity,
         void* custom_workspace,
         size_t* custom_workspace_size,
         bool opg)
{
  ASSERT(!opg, "DBSCAN for multi-groups doesn't support multi-GPU");
  dbscanFitImpl<double, int, false>(handle,
                                    input,
                                    n_groups,
                                    n_rows_ptr,
                                    n_cols,
                                    eps_ptr,
                                    min_pts_ptr,
                                    metric,
                                    labels,
                                    core_sample_indices,
                                    max_bytes_per_batch / 1e6,
                                    handle.get_stream(),
                                    verbosity,
                                    custom_workspace,
                                    custom_workspace_size);
}

void fit(const raft::handle_t& handle,
         float* input,
         int64_t n_groups,
         int64_t* n_rows_ptr,
         int64_t n_cols,
         const float* eps_ptr,
         const int64_t* min_pts_ptr,
         raft::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices,
         size_t max_bytes_per_batch,
         int verbosity,
         void* custom_workspace,
         size_t* custom_workspace_size,
         bool opg)
{
  ASSERT(!opg, "DBSCAN for multi-groups doesn't support multi-GPU");
  dbscanFitImpl<float, int64_t, false>(handle,
                                       input,
                                       n_groups,
                                       n_rows_ptr,
                                       n_cols,
                                       eps_ptr,
                                       min_pts_ptr,
                                       metric,
                                       labels,
                                       core_sample_indices,
                                       max_bytes_per_batch / 1e6,
                                       handle.get_stream(),
                                       verbosity,
                                       custom_workspace,
                                       custom_workspace_size);
}

void fit(const raft::handle_t& handle,
         double* input,
         int64_t n_groups,
         int64_t* n_rows_ptr,
         int64_t n_cols,
         const double* eps_ptr,
         const int64_t* min_pts_ptr,
         raft::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices,
         size_t max_bytes_per_batch,
         int verbosity,
         void* custom_workspace,
         size_t* custom_workspace_size,
         bool opg)
{
  ASSERT(!opg, "DBSCAN for multi-groups doesn't support multi-GPU");
  dbscanFitImpl<double, int64_t, false>(handle,
                                        input,
                                        n_groups,
                                        n_rows_ptr,
                                        n_cols,
                                        eps_ptr,
                                        min_pts_ptr,
                                        metric,
                                        labels,
                                        core_sample_indices,
                                        max_bytes_per_batch / 1e6,
                                        handle.get_stream(),
                                        verbosity,
                                        custom_workspace,
                                        custom_workspace_size);
}

}  // namespace Dbscan
}  // namespace ML
