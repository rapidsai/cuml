/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

#include "dbscan.cuh"

#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/distance_type.hpp>

#include <raft/util/cudart_utils.hpp>

namespace ML {
namespace Dbscan {

void fit(const raft::handle_t& handle,
         float* input,
         int n_rows,
         int n_cols,
         float eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices,
         float* sample_weight,
         size_t max_bytes_per_batch,
         EpsNnMethod eps_nn_method,
         rapids_logger::level_enum verbosity,
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
                                    sample_weight,
                                    max_bytes_per_batch,
                                    eps_nn_method,
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
                                     sample_weight,
                                     max_bytes_per_batch,
                                     eps_nn_method,
                                     handle.get_stream(),
                                     verbosity);
}

void fit(const raft::handle_t& handle,
         double* input,
         int n_rows,
         int n_cols,
         double eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices,
         double* sample_weight,
         size_t max_bytes_per_batch,
         EpsNnMethod eps_nn_method,
         rapids_logger::level_enum verbosity,
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
                                     sample_weight,
                                     max_bytes_per_batch,
                                     eps_nn_method,
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
                                      sample_weight,
                                      max_bytes_per_batch,
                                      eps_nn_method,
                                      handle.get_stream(),
                                      verbosity);
}

void fit(const raft::handle_t& handle,
         float* input,
         int64_t n_rows,
         int64_t n_cols,
         float eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices,
         float* sample_weight,
         size_t max_bytes_per_batch,
         EpsNnMethod eps_nn_method,
         rapids_logger::level_enum verbosity,
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
                                        sample_weight,
                                        max_bytes_per_batch,
                                        eps_nn_method,
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
                                         sample_weight,
                                         max_bytes_per_batch,
                                         eps_nn_method,
                                         handle.get_stream(),
                                         verbosity);
}

void fit(const raft::handle_t& handle,
         double* input,
         int64_t n_rows,
         int64_t n_cols,
         double eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices,
         double* sample_weight,
         size_t max_bytes_per_batch,
         EpsNnMethod eps_nn_method,
         rapids_logger::level_enum verbosity,
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
                                         sample_weight,
                                         max_bytes_per_batch,
                                         eps_nn_method,
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
                                          sample_weight,
                                          max_bytes_per_batch,
                                          eps_nn_method,
                                          handle.get_stream(),
                                          verbosity);
}

}  // namespace Dbscan
}  // namespace ML
