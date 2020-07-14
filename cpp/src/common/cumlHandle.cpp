/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <common/cudart_utils.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/cusolver_wrappers.h>
#include <sparse/cusparse_wrappers.h>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/logger.hpp>
#include "handle_impl.hpp"
#include "raftHandle_impl.hpp"

namespace ML {

using MLCommon::deviceAllocator;
using MLCommon::hostAllocator;

int cumlHandle::getDefaultNumInternalStreams() {
  return _default_num_internal_streams;
}

cumlHandle::cumlHandle(int n_streams) : _impl(new raftHandle_impl(n_streams)) {}
cumlHandle::cumlHandle(raft::handle_t* raftHandle) : _impl(new raftHandle_impl(raftHandle)) {}
cumlHandle::cumlHandle() : _impl(new raftHandle_impl()) {}
cumlHandle::~cumlHandle() {}

void cumlHandle::setStream(cudaStream_t stream) { _impl->setStream(stream); }

cudaStream_t cumlHandle::getStream() const { return _impl->getStream(); }

const cudaDeviceProp& cumlHandle::getDeviceProperties() const {
  return _impl->getDeviceProperties();
}

std::vector<cudaStream_t> cumlHandle::getInternalStreams() const {
  return _impl->getInternalStreams();
}

void cumlHandle::setDeviceAllocator(
  std::shared_ptr<deviceAllocator> allocator) {
  _impl->setDeviceAllocator(allocator);
}

std::shared_ptr<deviceAllocator> cumlHandle::getDeviceAllocator() const {
  return _impl->getDeviceAllocator();
}

void cumlHandle::setHostAllocator(std::shared_ptr<hostAllocator> allocator) {
  _impl->setHostAllocator(allocator);
}

std::shared_ptr<hostAllocator> cumlHandle::getHostAllocator() const {
  return _impl->getHostAllocator();
}
int cumlHandle::getNumInternalStreams() {
  return _impl->getNumInternalStreams();
}
const handle_impl& cumlHandle::getImpl() const { return *_impl.get(); }

handle_impl& cumlHandle::getImpl() { return *_impl.get(); }

HandleMap handleMap;

std::pair<cumlHandle_t, cumlError_t> HandleMap::createAndInsertHandle() {
  cumlError_t status = CUML_SUCCESS;
  cumlHandle_t chosen_handle;
  try {
    auto handle_ptr = new ML::cumlHandle();
    bool inserted;
    {
      std::lock_guard<std::mutex> guard(_mapMutex);
      cumlHandle_t initial_next = _nextHandle;
      do {
        // try to insert using next free handle identifier
        chosen_handle = _nextHandle;
        inserted = _handleMap.insert({chosen_handle, handle_ptr}).second;
        _nextHandle += 1;
      } while (!inserted && _nextHandle != initial_next);
    }
    if (!inserted) {
      // no free handle identifier available
      chosen_handle = INVALID_HANDLE;
      status = CUML_ERROR_UNKNOWN;
    }
  }
  //TODO: Implement this
  //catch (const MLCommon::Exception& e)
  //{
  //    //log e.what()?
  //    status =  e.getErrorCode();
  //}
  catch (...) {
    status = CUML_ERROR_UNKNOWN;
    chosen_handle = CUML_ERROR_UNKNOWN;
  }
  return std::pair<cumlHandle_t, cumlError_t>(chosen_handle, status);
}

std::pair<cumlHandle*, cumlError_t> HandleMap::lookupHandlePointer(
  cumlHandle_t handle) const {
  std::lock_guard<std::mutex> guard(_mapMutex);
  auto it = _handleMap.find(handle);
  if (it == _handleMap.end()) {
    return std::pair<cumlHandle*, cumlError_t>(nullptr, CUML_INVALID_HANDLE);
  } else {
    return std::pair<cumlHandle*, cumlError_t>(it->second, CUML_SUCCESS);
  }
}

cumlError_t HandleMap::removeAndDestroyHandle(cumlHandle_t handle) {
  ML::cumlHandle* handle_ptr;
  {
    std::lock_guard<std::mutex> guard(_mapMutex);
    auto it = _handleMap.find(handle);
    if (it == _handleMap.end()) {
      return CUML_INVALID_HANDLE;
    }
    handle_ptr = it->second;
    _handleMap.erase(it);
  }
  cumlError_t status = CUML_SUCCESS;
  try {
    delete handle_ptr;
  }
  //TODO: Implement this
  //catch (const MLCommon::Exception& e)
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
