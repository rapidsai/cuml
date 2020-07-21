/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <common/cuml_comms_int.hpp>

#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/raftAllocatorAdapter.hpp>

#include <raft/handle.hpp>

namespace ML {

using MLCommon::deviceAllocator;
using MLCommon::hostAllocator;
using MLCommon::raftHostAllocatorAdapter;

/**
 * @todo: Add doxygen documentation
 */
class cumlHandle_impl : raft::handle_t {
 public:
  cumlHandle_impl(int n_streams = cumlHandle::getDefaultNumInternalStreams())
    : raft::handle_t(n_streams) { }
  ~cumlHandle_impl() { }

  int getDevice() const { return raft::handle_t::get_device(); }

  void setStream(cudaStream_t stream) { raft::handle_t::set_stream(stream); }

  cudaStream_t getStream() const { return raft::handle_t::get_stream(); }

  const cudaDeviceProp& getDeviceProperties() const {
    return raft::handle_t::get_device_properties();
  }

  void setDeviceAllocator(std::shared_ptr<deviceAllocator> allocator) {
    raft::handle_t::set_device_allocator(allocator->getRaftDeviceAllocator());
  }

  std::shared_ptr<deviceAllocator> getDeviceAllocator() const {
    if(!_deviceAllocatorInitialized) {
      _deviceAllocator = std::make_shared<raftDeviceAllocatorAdapter>(raft::handle_t::get_device_allocator());
      _deviceAllocatorInitialized = true;
    }
    return _deviceAllocator;
  }

  void setHostAllocator(std::shared_ptr<hostAllocator> allocator) {
    raft::handle_t::set_device_allocator(allocator->getRaftHostAllocator());
  }

  std::shared_ptr<hostAllocator> getHostAllocator() const {
    if(!_hostAllocatorInitialized) {
      _hostAllocator = std::make_shared<raftHostAllocatorAdapter>(raft::handle_t::get_host_allocator());
      _hostAllocatorInitialized = true;
    }
    return _hostAllocator;
  }

  cublasHandle_t getCublasHandle() const {
    return raft::handle_t::get_cublas_handle();
  }

  cusolverDnHandle_t getcusolverDnHandle() const {
    return raft::handle_t::get_cusolver_dn_handle();
  }

  cusolverSpHandle_t getcusolverSpHandle() const {
    return raft::handle_t::get_cusolver_sp_handle();
  }

  cusparseHandle_t getcusparseHandle() const {
    return raft::handle_t::get_cusparse_handle();
  }

  cudaStream_t getInternalStream(int sid) const {
    return raft::handle_t::get_internal_stream(sid);
  }

  int getNumInternalStreams() const {
    return raft::handle_t::get_num_internal_streams();
  }

  std::vector<cudaStream_t> getInternalStreams() const {
    return raft::handle_t::get_internal_streams();
  }

  void waitOnUserStream() const {
    raft::handle_t::wait_on_user_stream();
  }
  void waitOnInternalStreams() const {
    raft::handle_t::wait_on_internal_streams();
  }

  void setCommunicator(
    std::shared_ptr<MLCommon::cumlCommunicator> communicator) {
      raft::handle_t::set_comms(communicator);
  }

  const MLCommon::cumlCommunicator& getCommunicator() const {
    return dynamic_cast<const MLCommon::cumlCommunicator&>(raft::handle_t::get_comms());
  }

  bool commsInitialized() const {
    return raft::handle_t::comms_initialized();
  }

  private:
    bool _hostAllocatorInitialized = false;
    bool _deviceAllocatorInitialized = false;

    std::shared_ptr<deviceAllocator> _deviceAllocator;
    std::shared_ptr<hostAllocator> _hostAllocator

};

}