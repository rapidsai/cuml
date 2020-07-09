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

namespace ML {

using MLCommon::deviceAllocator;
using MLCommon::hostAllocator;

class handle_impl {
 public:
  handle_impl(int n_streams = cumlHandle::getDefaultNumInternalStreams()) {}
  virtual ~handle_impl() {}

  virtual int getDevice() const = 0;
  virtual void setStream(cudaStream_t stream) = 0;
  virtual cudaStream_t getStream() const = 0;
  virtual void setDeviceAllocator(
    std::shared_ptr<deviceAllocator> allocator) = 0;
  virtual std::shared_ptr<deviceAllocator> getDeviceAllocator() const = 0;
  virtual void setHostAllocator(std::shared_ptr<hostAllocator> allocator) = 0;
  virtual std::shared_ptr<hostAllocator> getHostAllocator() const = 0;

  virtual cublasHandle_t getCublasHandle() const = 0;
  virtual cusolverDnHandle_t getcusolverDnHandle() const = 0;
  virtual cusolverSpHandle_t getcusolverSpHandle() const = 0;
  virtual cusparseHandle_t getcusparseHandle() const = 0;

  virtual cudaStream_t getInternalStream(int sid) const = 0;
  virtual int getNumInternalStreams() const = 0;

  virtual std::vector<cudaStream_t> getInternalStreams() const = 0;

  virtual void waitOnUserStream() const = 0;
  virtual void waitOnInternalStreams() const = 0;

  virtual void setCommunicator(
    std::shared_ptr<MLCommon::cumlCommunicator> communicator) = 0;
  virtual const MLCommon::cumlCommunicator& getCommunicator() const = 0;
  virtual bool commsInitialized() const = 0;

  virtual const cudaDeviceProp& getDeviceProperties() const = 0;
};

}  // end namespace ML