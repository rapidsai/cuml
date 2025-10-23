/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
