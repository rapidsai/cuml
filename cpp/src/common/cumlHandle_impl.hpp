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
   cumlHandle_impl(int n_streams = cumlHandle::getDefaultNumInternalStreams());
  ~cumlHandle_impl();

  int getDevice() const;
  void setStream(cudaStream_t stream);
  cudaStream_t getStream() const;
  void setDeviceAllocator(std::shared_ptr<deviceAllocator> allocator);
  std::shared_ptr<deviceAllocator> getDeviceAllocator() const;
  void setHostAllocator(std::shared_ptr<hostAllocator> allocator);
  std::shared_ptr<hostAllocator> getHostAllocator() const;

  cublasHandle_t getCublasHandle() const;
  cusolverDnHandle_t getcusolverDnHandle() const;
  cusolverSpHandle_t getcusolverSpHandle() const;
  cusparseHandle_t getcusparseHandle() const;

  cudaStream_t getInternalStream(int sid) const;
  int getNumInternalStreams() const;

  std::vector<cudaStream_t> getInternalStreams() const;

  void waitOnUserStream() const;
  void waitOnInternalStreams() const;

  void setCommunicator(
    std::shared_ptr<MLCommon::cumlCommunicator> communicator);
  const MLCommon::cumlCommunicator& getCommunicator() const;
  bool commsInitialized() const;

  const cudaDeviceProp& getDeviceProperties() const;

 private:
  mutable bool _hostAllocatorInitialized = false;
  mutable bool _deviceAllocatorInitialized = false;

  mutable std::shared_ptr<deviceAllocator> _deviceAllocator;
  mutable std::shared_ptr<hostAllocator> _hostAllocator;
};

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

}  // namespace ML