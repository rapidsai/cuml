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

#include <common/cumlHandle.hpp>

#include <cuml/cluster/dbscan.hpp>
#include <cuml/cluster/dbscan_api.h>
#include <cuml/common/distance_type.hpp>
#include <cuml/cuml_api.h>

extern "C" {

cumlError_t cumlSpDbscanFit(cumlHandle_t handle,
                            float* input,
                            int n_rows,
                            int n_cols,
                            float eps,
                            int min_pts,
                            int* labels,
                            int* core_sample_indices,
                            size_t max_bytes_per_batch,
                            int verbosity)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::Dbscan::fit(*handle_ptr,
                      input,
                      n_rows,
                      n_cols,
                      eps,
                      min_pts,
                      ML::distance::DistanceType::L2SqrtUnexpanded,
                      labels,
                      core_sample_indices,
                      NULL,
                      max_bytes_per_batch,
                      ML::Dbscan::EpsNnMethod::BRUTE_FORCE,
                      static_cast<rapids_logger::level_enum>(verbosity));
    }
    // TODO: Implement this
    // catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlDpDbscanFit(cumlHandle_t handle,
                            double* input,
                            int n_rows,
                            int n_cols,
                            double eps,
                            int min_pts,
                            int* labels,
                            int* core_sample_indices,
                            size_t max_bytes_per_batch,
                            int verbosity)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::Dbscan::fit(*handle_ptr,
                      input,
                      n_rows,
                      n_cols,
                      eps,
                      min_pts,
                      ML::distance::DistanceType::L2SqrtUnexpanded,
                      labels,
                      core_sample_indices,
                      NULL,
                      max_bytes_per_batch,
                      ML::Dbscan::EpsNnMethod::BRUTE_FORCE,
                      static_cast<rapids_logger::level_enum>(verbosity));
    }
    // TODO: Implement this
    // catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}
}
