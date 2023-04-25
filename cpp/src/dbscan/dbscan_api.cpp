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

#include <cuml/cluster/dbscan_api.h>

#include <common/cumlHandle.hpp>
#include <cuml/cluster/dbscan.hpp>
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
                      raft::distance::L2SqrtUnexpanded,
                      labels,
                      core_sample_indices,
                      max_bytes_per_batch,
                      verbosity);
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
                      raft::distance::L2SqrtUnexpanded,
                      labels,
                      core_sample_indices,
                      max_bytes_per_batch,
                      verbosity);
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

cumlError_t cumlMultiSpDbscanFit(cumlHandle_t handle,
                                 float* input,
                                 int n_groups,
                                 int* n_rows_ptr,
                                 int n_cols,
                                 const float* eps_ptr,
                                 const int* min_pts_ptr,
                                 int* labels,
                                 int* core_sample_indices,
                                 size_t max_bytes_per_batch,
                                 int verbosity,
                                 void* custom_workspace,
                                 size_t* custom_workspace_size)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::Dbscan::fit(*handle_ptr,
                      input,
                      n_groups,
                      n_rows_ptr,
                      n_cols,
                      eps_ptr,
                      min_pts_ptr,
                      raft::distance::L2SqrtUnexpanded,
                      labels,
                      core_sample_indices,
                      max_bytes_per_batch,
                      verbosity,
                      custom_workspace,
                      custom_workspace_size);
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

cumlError_t cumlMultiDpDbscanFit(cumlHandle_t handle,
                                 double* input,
                                 int n_groups,
                                 int* n_rows_ptr,
                                 int n_cols,
                                 const double* eps_ptr,
                                 const int* min_pts_ptr,
                                 int* labels,
                                 int* core_sample_indices,
                                 size_t max_bytes_per_batch,
                                 int verbosity,
                                 void* custom_workspace,
                                 size_t* custom_workspace_size)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::Dbscan::fit(*handle_ptr,
                      input,
                      n_groups,
                      n_rows_ptr,
                      n_cols,
                      eps_ptr,
                      min_pts_ptr,
                      raft::distance::L2SqrtUnexpanded,
                      labels,
                      core_sample_indices,
                      max_bytes_per_batch,
                      verbosity,
                      custom_workspace,
                      custom_workspace_size);
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