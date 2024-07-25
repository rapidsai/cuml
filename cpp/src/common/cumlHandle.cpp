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

#include "cumlHandle.hpp"

#include <cuml/common/logger.hpp>

#include <raft/util/cudart_utils.hpp>
// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cublas_wrappers.hpp>
// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cusolver_wrappers.hpp>

namespace ML {

HandleMap handleMap;

std::pair<cumlHandle_t, cumlError_t> HandleMap::createAndInsertHandle(cudaStream_t stream)
{
  cumlError_t status = CUML_SUCCESS;
  cumlHandle_t chosen_handle;
  try {
    auto handle_ptr = new raft::handle_t{stream};
    bool inserted;
    {
      std::lock_guard<std::mutex> guard(_mapMutex);
      cumlHandle_t initial_next = _nextHandle;
      do {
        // try to insert using next free handle identifier
        chosen_handle = _nextHandle;
        inserted      = _handleMap.insert({chosen_handle, handle_ptr}).second;
        _nextHandle += 1;
      } while (!inserted && _nextHandle != initial_next);
    }
    if (!inserted) {
      // no free handle identifier available
      chosen_handle = INVALID_HANDLE;
      status        = CUML_ERROR_UNKNOWN;
    }
  }
  // TODO: Implement this
  // catch (const MLCommon::Exception& e)
  //{
  //    //log e.what()?
  //    status =  e.getErrorCode();
  //}
  catch (...) {
    status        = CUML_ERROR_UNKNOWN;
    chosen_handle = CUML_ERROR_UNKNOWN;
  }
  return std::pair<cumlHandle_t, cumlError_t>(chosen_handle, status);
}

std::pair<raft::handle_t*, cumlError_t> HandleMap::lookupHandlePointer(cumlHandle_t handle) const
{
  std::lock_guard<std::mutex> guard(_mapMutex);
  auto it = _handleMap.find(handle);
  if (it == _handleMap.end()) {
    return std::pair<raft::handle_t*, cumlError_t>(nullptr, CUML_INVALID_HANDLE);
  } else {
    return std::pair<raft::handle_t*, cumlError_t>(it->second, CUML_SUCCESS);
  }
}

cumlError_t HandleMap::removeAndDestroyHandle(cumlHandle_t handle)
{
  raft::handle_t* handle_ptr;
  {
    std::lock_guard<std::mutex> guard(_mapMutex);
    auto it = _handleMap.find(handle);
    if (it == _handleMap.end()) { return CUML_INVALID_HANDLE; }
    handle_ptr = it->second;
    _handleMap.erase(it);
  }
  cumlError_t status = CUML_SUCCESS;
  try {
    delete handle_ptr;
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
  return status;
}

}  // end namespace ML
