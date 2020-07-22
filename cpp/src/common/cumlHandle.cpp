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

namespace ML {

using MLCommon::deviceAllocator;
using MLCommon::hostAllocator;

int cumlHandle::getDefaultNumInternalStreams() {
  return _default_num_internal_streams;
}

cumlHandle::cumlHandle(int n_streams) : _impl(new cumlHandle_impl(n_streams)) {}
cumlHandle::cumlHandle() : _impl(new cumlHandle_impl()) {}
cumlHandle::cumlHandle(raft::handle_t* raftHandle) {
  _impl = std::unique_ptr<cumlHandle_impl>(
    dynamic_cast<cumlHandle_impl*>(raftHandle));
}
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
const cumlHandle_impl& cumlHandle::getImpl() const { return *_impl.get(); }

cumlHandle_impl& cumlHandle::getImpl() { return *_impl.get(); }

cumlHandle_impl::cumlHandle_impl(int n_streams)
    : raft::handle_t(n_streams) {}
  cumlHandle_impl::~cumlHandle_impl() {}

  int cumlHandle_impl::getDevice() const { return raft::handle_t::get_device(); }

  void cumlHandle_impl::setStream(cudaStream_t stream) { raft::handle_t::set_stream(stream); }

  cudaStream_t cumlHandle_impl::getStream() const { return raft::handle_t::get_stream(); }

  const cudaDeviceProp& cumlHandle_impl::getDeviceProperties() const {
    return raft::handle_t::get_device_properties();
  }

  void cumlHandle_impl::setDeviceAllocator(std::shared_ptr<deviceAllocator> allocator) {
    raft::handle_t::set_device_allocator(
      std::dynamic_pointer_cast<raftDeviceAllocatorAdapter>(allocator)
        ->getRaftDeviceAllocator());
  }

  std::shared_ptr<deviceAllocator> cumlHandle_impl::getDeviceAllocator() const {
    if (!_deviceAllocatorInitialized) {
      _deviceAllocatorInitialized = true;
    }
    return _deviceAllocator;
  }

  void cumlHandle_impl::setHostAllocator(std::shared_ptr<hostAllocator> allocator) {
    raft::handle_t::set_host_allocator(
      std::dynamic_pointer_cast<raftHostAllocatorAdapter>(allocator)
        ->getRaftHostAllocator());
  }

  std::shared_ptr<hostAllocator> cumlHandle_impl::getHostAllocator() const {
    if (!_hostAllocatorInitialized) {
      _hostAllocator = std::make_shared<raftHostAllocatorAdapter>(
        raft::handle_t::get_host_allocator());
      _hostAllocatorInitialized = true;
    }
    return _hostAllocator;
  }

  cublasHandle_t cumlHandle_impl::getCublasHandle() const {
    return raft::handle_t::get_cublas_handle();
  }

  cusolverDnHandle_t cumlHandle_impl::getcusolverDnHandle() const {
    return raft::handle_t::get_cusolver_dn_handle();
  }

  cusolverSpHandle_t cumlHandle_impl::getcusolverSpHandle() const {
    return raft::handle_t::get_cusolver_sp_handle();
  }

  cusparseHandle_t cumlHandle_impl::getcusparseHandle() const {
    return raft::handle_t::get_cusparse_handle();
  }

  cudaStream_t cumlHandle_impl::getInternalStream(int sid) const {
    return raft::handle_t::get_internal_stream(sid);
  }

  int cumlHandle_impl::getNumInternalStreams() const {
    return raft::handle_t::get_num_internal_streams();
  }

  std::vector<cudaStream_t> cumlHandle_impl::getInternalStreams() const {
    return raft::handle_t::get_internal_streams();
  }

  void cumlHandle_impl::waitOnUserStream() const { raft::handle_t::wait_on_user_stream(); }
  void cumlHandle_impl::waitOnInternalStreams() const {
    raft::handle_t::wait_on_internal_streams();
  }

  void cumlHandle_impl::setCommunicator(
    std::shared_ptr<MLCommon::cumlCommunicator> communicator) {
    raft::handle_t::set_comms(communicator);
  }

  const MLCommon::cumlCommunicator& cumlHandle_impl::getCommunicator() const {
    return dynamic_cast<const MLCommon::cumlCommunicator&>(
      raft::handle_t::get_comms());
  }

  bool cumlHandle_impl::commsInitialized() const { return raft::handle_t::comms_initialized(); }

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

namespace detail {

streamSyncer::streamSyncer(const cumlHandle_impl& handle) : _handle(handle) {
  _handle.waitOnUserStream();
}
streamSyncer::~streamSyncer() { _handle.waitOnInternalStreams(); }

}  // namespace detail

}  // end namespace ML
