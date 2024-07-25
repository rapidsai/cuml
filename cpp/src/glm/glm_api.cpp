/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cuml/linear_model/glm.hpp>
#include <cuml/linear_model/glm_api.h>
#include <cuml/linear_model/qn.h>

namespace ML::GLM {

extern "C" {

cumlError_t cumlSpQnFit(cumlHandle_t cuml_handle,
                        const qn_params* pams,
                        float* X,
                        float* y,
                        int N,
                        int D,
                        int C,
                        float* w0,
                        float* f,
                        int* num_iters,
                        bool X_col_major)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(cuml_handle);
  if (status == CUML_SUCCESS) {
    try {
      qnFit(*handle_ptr, *pams, X, X_col_major, y, N, D, C, w0, f, num_iters);
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

cumlError_t cumlDpQnFit(cumlHandle_t cuml_handle,
                        const qn_params* pams,
                        double* X,
                        double* y,
                        int N,
                        int D,
                        int C,
                        double* w0,
                        double* f,
                        int* num_iters,
                        bool X_col_major)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(cuml_handle);
  if (status == CUML_SUCCESS) {
    try {
      qnFit(*handle_ptr, *pams, X, X_col_major, y, N, D, C, w0, f, num_iters);

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
}  // namespace ML::GLM
