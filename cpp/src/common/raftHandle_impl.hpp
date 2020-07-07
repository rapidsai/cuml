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

#pragma once

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <common/cuml_comms_int.hpp>

#include <cuml/cuml.hpp>

#include <cuml/common/cuml_allocator.hpp>

#include <raft/handle.hpp>
#include "cumlHandle.hpp"

namespace ML {

using MLCommon::deviceAllocator;
using MLCommon::hostAllocator;

/**
 * @todo: Add doxygen documentation
 */
class raftHandle_impl : public cumlHandle_impl{
 public:
  raftHandle_impl(int n_streams = cumlHandle::getDefaultNumInternalStreams());
  ~raftHandle_impl();

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

  raft::handle_t& getRaftHandle();

 private:
  std::shared_ptr<deviceAllocator> _deviceAllocator;
  std::shared_ptr<hostAllocator> _hostAllocator;
  std::shared_ptr<MLCommon::cumlCommunicator> _communicator;

  raft::handle_t* _raftHandle;

};

} // end namespace ML
