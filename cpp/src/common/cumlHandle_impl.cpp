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

#include "cumlHandle_impl.hpp"

#include <common/cudart_utils.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/cusolver_wrappers.h>
#include <sparse/cusparse_wrappers.h>

#include <cuml/common/cuml_allocator.hpp>

using MLCommon::defaultDeviceAllocator;
using MLCommon::defaultHostAllocator;

namespace ML {

cumlHandle_impl::cumlHandle_impl(int n_streams)
  : _cublas_handle(),
    _cusolverDn_handle(),
    _cusolverSp_handle(),
    _cusparse_handle(),
    _userStream(nullptr),
    _event(),
    _deviceAllocator(std::make_shared<defaultDeviceAllocator>()),
    _hostAllocator(std::make_shared<defaultHostAllocator>()),
    _communicator(),
    _streams(),
    _prop(),
    _dev_id([]() -> int {
      int cur_dev = -1;
      CUDA_CHECK(cudaGetDevice(&cur_dev));
      return cur_dev;
    }()),
    _num_streams(n_streams),
    _cublasInitialized(false),
    _cusolverDnInitialized(false),
    _cusolverSpInitialized(false),
    _cusparseInitialized(false),
    _devicePropInitialized(false) {
  createResources();
}

cumlHandle_impl::~cumlHandle_impl() { destroyResources(); }

int cumlHandle_impl::getDevice() const { return _dev_id; }

void cumlHandle_impl::setStream(cudaStream_t stream) { _userStream = stream; }

cudaStream_t cumlHandle_impl::getStream() const { return _userStream; }

const cudaDeviceProp& cumlHandle_impl::getDeviceProperties() const {
  if (!_devicePropInitialized) {
    CUDA_CHECK(cudaGetDeviceProperties(&_prop, _dev_id));
    _devicePropInitialized = true;
  }
  return _prop;
}

void cumlHandle_impl::setDeviceAllocator(
  std::shared_ptr<deviceAllocator> allocator) {
  _deviceAllocator = allocator;
}

std::shared_ptr<deviceAllocator> cumlHandle_impl::getDeviceAllocator() const {
  return _deviceAllocator;
}

void cumlHandle_impl::setHostAllocator(
  std::shared_ptr<hostAllocator> allocator) {
  _hostAllocator = allocator;
}

std::shared_ptr<hostAllocator> cumlHandle_impl::getHostAllocator() const {
  return _hostAllocator;
}

cublasHandle_t cumlHandle_impl::getCublasHandle() const {
  if (!_cublasInitialized) {
    CUBLAS_CHECK(cublasCreate(&_cublas_handle));
    _cublasInitialized = true;
  }
  return _cublas_handle;
}

cusolverDnHandle_t cumlHandle_impl::getcusolverDnHandle() const {
  if (!_cusolverDnInitialized) {
    CUSOLVER_CHECK(cusolverDnCreate(&_cusolverDn_handle));
    _cusolverDnInitialized = true;
  }
  return _cusolverDn_handle;
}

cusolverSpHandle_t cumlHandle_impl::getcusolverSpHandle() const {
  if (!_cusolverSpInitialized) {
    CUSOLVER_CHECK(cusolverSpCreate(&_cusolverSp_handle));
    _cusolverSpInitialized = true;
  }
  return _cusolverSp_handle;
}

cusparseHandle_t cumlHandle_impl::getcusparseHandle() const {
  if (!_cusparseInitialized) {
    CUSPARSE_CHECK(cusparseCreate(&_cusparse_handle));
    _cusparseInitialized = true;
  }
  return _cusparse_handle;
}

cudaStream_t cumlHandle_impl::getInternalStream(int sid) const {
  return _streams[sid];
}

int cumlHandle_impl::getNumInternalStreams() const { return _num_streams; }

std::vector<cudaStream_t> cumlHandle_impl::getInternalStreams() const {
  std::vector<cudaStream_t> int_streams_vec(_num_streams);
  for (auto s : _streams) {
    int_streams_vec.push_back(s);
  }
  return int_streams_vec;
}

void cumlHandle_impl::waitOnUserStream() const {
  CUDA_CHECK(cudaEventRecord(_event, _userStream));
  for (auto s : _streams) {
    CUDA_CHECK(cudaStreamWaitEvent(s, _event, 0));
  }
}

void cumlHandle_impl::waitOnInternalStreams() const {
  for (auto s : _streams) {
    CUDA_CHECK(cudaEventRecord(_event, s));
    CUDA_CHECK(cudaStreamWaitEvent(_userStream, _event, 0));
  }
}

void cumlHandle_impl::setCommunicator(
  std::shared_ptr<MLCommon::cumlCommunicator> communicator) {
  _communicator = communicator;
}

const MLCommon::cumlCommunicator& cumlHandle_impl::getCommunicator() const {
  ASSERT(nullptr != _communicator.get(),
         "ERROR: Communicator was not initialized\n");
  return *_communicator;
}

bool cumlHandle_impl::commsInitialized() const {
  return (nullptr != _communicator.get());
}

void cumlHandle_impl::createResources() {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  _streams.push_back(stream);
  for (int i = 1; i < _num_streams; ++i) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    _streams.push_back(stream);
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming));
}

void cumlHandle_impl::destroyResources() {
  if (_cusparseInitialized) {
    CUSPARSE_CHECK_NO_THROW(cusparseDestroy(_cusparse_handle));
  }
  if (_cusolverDnInitialized) {
    CUSOLVER_CHECK_NO_THROW(cusolverDnDestroy(_cusolverDn_handle));
  }
  if (_cusolverSpInitialized) {
    CUSOLVER_CHECK_NO_THROW(cusolverSpDestroy(_cusolverSp_handle));
  }
  if (_cublasInitialized) {
    CUBLAS_CHECK_NO_THROW(cublasDestroy(_cublas_handle));
  }
  while (!_streams.empty()) {
    CUDA_CHECK_NO_THROW(cudaStreamDestroy(_streams.back()));
    _streams.pop_back();
  }
  CUDA_CHECK_NO_THROW(cudaEventDestroy(_event));
}

}  // end namespace ML