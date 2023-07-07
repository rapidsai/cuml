/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
