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

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include "common/cuml_comms_int.hpp"

#include "../cuML.hpp"
#include "../cuML_api.h"

namespace ML {

/**
 * @todo: Add doxygen documentation
 */
class cumlHandle_impl {
 public:
  cumlHandle_impl();
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
  cusparseHandle_t getcusparseHandle() const;

  cudaStream_t getInternalStream(int sid) const;
  int getNumInternalStreams() const;

  void waitOnUserStream() const;
  void waitOnInternalStreams() const;

  void setCommunicator(
    std::shared_ptr<MLCommon::cumlCommunicator> communicator);
  const MLCommon::cumlCommunicator& getCommunicator() const;
  bool commsInitialized() const;

  const cudaDeviceProp& getDeviceProp() const;

 private:
  //TODO: What is the right number?
  static constexpr int _num_streams = 3;
  const int _dev_id;
  std::vector<cudaStream_t> _streams;
  cublasHandle_t _cublas_handle;
  cusolverDnHandle_t _cusolverDn_handle;
  cusparseHandle_t _cusparse_handle;
  std::shared_ptr<deviceAllocator> _deviceAllocator;
  std::shared_ptr<hostAllocator> _hostAllocator;
  cudaStream_t _userStream;
  cudaEvent_t _event;
  cudaDeviceProp prop;

  std::shared_ptr<MLCommon::cumlCommunicator> _communicator;

  void createResources();
  void destroyResources();
};

/**
 * Map from integral cumlHandle_t identifiers to cumlHandle pointer protected
 * by a mutex for thread-safe access.
 */
class HandleMap {
 public:
  /**
     * @brief Creates new handle object with associated handle ID and insert into map.
     *
     * @return std::pair with handle and error code. If error code is not CUML_SUCCESS
     *                   the handle is INVALID_HANDLE.
     */
  std::pair<cumlHandle_t, cumlError_t> createAndInsertHandle();

  /**
     * @brief Lookup pointer to handle object for handle ID in map.
     *
     * @return std::pair with handle and error code. If error code is not CUML_SUCCESS
     *                   the handle is INVALID_HANDLE. Error code CUML_INAVLID_HANDLE
     *                   is returned if the provided `handle` is invald.
     */
  std::pair<cumlHandle*, cumlError_t> lookupHandlePointer(
    cumlHandle_t handle) const;

  /**
     * @brief Remove handle from map and destroy associated handle object.
     *
     * @return cumlError_t CUML_SUCCESS or CUML_INVALID_HANDLE.
     *                   Error code CUML_INAVLID_HANDLE is returned if the provided
     *                   `handle` is invald.
     */
  cumlError_t removeAndDestroyHandle(cumlHandle_t handle);

  static const cumlHandle_t INVALID_HANDLE =
    -1;  //!< sentinel value for invalid ID

 private:
  std::unordered_map<cumlHandle_t, cumlHandle*>
    _handleMap;                  //!< map from ID to pointer
  mutable std::mutex _mapMutex;  //!< mutex protecting the map
  cumlHandle_t _nextHandle;      //!< value of next handle ID
};

/// Static handle map instance (see cumlHandle.cpp)
extern HandleMap handleMap;

namespace detail {

/**
 * @todo: Add doxygen documentation
 */
class streamSyncer {
 public:
  streamSyncer(const cumlHandle_impl& handle) : _handle(handle) {
    _handle.waitOnUserStream();
  }
  ~streamSyncer() { _handle.waitOnInternalStreams(); }

  streamSyncer(const streamSyncer& other) = delete;
  streamSyncer& operator=(const streamSyncer& other) = delete;

 private:
  const cumlHandle_impl& _handle;
};

}  // end namespace detail

}  // end namespace ML
