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

#include "raftHandle_impl.hpp"

#include <cuml/common/logger.hpp>
#include <cuml/common/rmmAllocatorAdapter.hpp>

namespace ML {

raftHandle_impl::raftHandle_impl(int n_streams) : 
    _hostAllocator(std::make_shared<defaultHostAllocator>()),
    _communicator() {
  _raftHandle = new raft::handle_t(n_streams);
  std::cout << n_streams << std::endl;
  std::cout << _raftHandle->get_num_internal_streams();

  _deviceAllocator = std::shared_ptr<rmmAllocatorAdapter>(
    new rmmAllocatorAdapter(_raftHandle->get_device_allocator()));
}

raftHandle_impl::~raftHandle_impl() { delete _raftHandle; }

int raftHandle_impl::getDevice() const { return _raftHandle->get_device(); }

void raftHandle_impl::setStream(cudaStream_t stream) {
  _raftHandle->set_stream(stream);
}

cudaStream_t raftHandle_impl::getStream() const {
  return _raftHandle->get_stream();
}

const cudaDeviceProp& raftHandle_impl::getDeviceProperties() const {
  return _raftHandle->get_device_properties();
}

void raftHandle_impl::setDeviceAllocator(
  std::shared_ptr<deviceAllocator> allocator) {
  _deviceAllocator = allocator;
}

std::shared_ptr<deviceAllocator> raftHandle_impl::getDeviceAllocator() const {
  return _deviceAllocator;
}

void raftHandle_impl::setHostAllocator(
  std::shared_ptr<hostAllocator> allocator) {
  _hostAllocator = allocator;
}

std::shared_ptr<hostAllocator> raftHandle_impl::getHostAllocator() const {
  return _hostAllocator;
}

cublasHandle_t raftHandle_impl::getCublasHandle() const {
  return _raftHandle->get_cublas_handle();
}

cusolverDnHandle_t raftHandle_impl::getcusolverDnHandle() const {
  return _raftHandle->get_cusolver_dn_handle();
}

cusolverSpHandle_t raftHandle_impl::getcusolverSpHandle() const {
  return _raftHandle->get_cusolver_sp_handle();
}

cusparseHandle_t raftHandle_impl::getcusparseHandle() const {
  return _raftHandle->get_cusparse_handle();
}

cudaStream_t raftHandle_impl::getInternalStream(int sid) const {
  return _raftHandle->get_internal_stream(sid);
}

int raftHandle_impl::getNumInternalStreams() const {
  std::cout << "HERE" << std::endl;
  std::cout << _raftHandle->get_num_internal_streams();
  return _raftHandle->get_num_internal_streams();
}

std::vector<cudaStream_t> raftHandle_impl::getInternalStreams() const {
  return _raftHandle->get_internal_streams();
}

void raftHandle_impl::waitOnUserStream() const {
  _raftHandle->wait_on_user_stream();
}

void raftHandle_impl::waitOnInternalStreams() const {
  _raftHandle->wait_on_internal_streams();
}

void raftHandle_impl::setCommunicator(
  std::shared_ptr<MLCommon::cumlCommunicator> communicator) {
  _communicator = communicator;
}

const MLCommon::cumlCommunicator& raftHandle_impl::getCommunicator() const {
  ASSERT(nullptr != _communicator.get(),
         "ERROR: Communicator was not initialized\n");
  return *_communicator;
}

bool raftHandle_impl::commsInitialized() const {
  return (nullptr != _communicator.get());
}

raft::handle_t& raftHandle_impl::getRaftHandle() { return *_raftHandle; }

}  // end namespace ML