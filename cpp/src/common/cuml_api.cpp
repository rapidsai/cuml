/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cumlHandle.hpp"

#include <cuml/common/utils.hpp>
#include <cuml/cuml_api.h>

#include <raft/util/cudart_utils.hpp>

#include <cstddef>
#include <functional>

extern "C" const char* cumlGetErrorString(cumlError_t error)
{
  switch (error) {
    case CUML_SUCCESS: return "success";
    case CUML_ERROR_UNKNOWN:
      // Intentional fall through
    default: return "unknown";
  }
}

extern "C" cumlError_t cumlCreate(cumlHandle_t* handle, cudaStream_t stream)
{
  cumlError_t status;
  std::tie(*handle, status) = ML::handleMap.createAndInsertHandle(stream);
  return status;
}

extern "C" cumlError_t cumlGetStream(cumlHandle_t handle, cudaStream_t* stream)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      *stream = handle_ptr->get_stream();
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

extern "C" cumlError_t cumlDestroy(cumlHandle_t handle)
{
  return ML::handleMap.removeAndDestroyHandle(handle);
}
