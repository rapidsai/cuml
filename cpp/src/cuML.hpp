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

#include <memory>

#include <cuda_runtime.h>

#include <common/cuml_allocator.hpp>

namespace ML {

using MLCommon::deviceAllocator;
using MLCommon::hostAllocator;

class cumlHandle_impl;

/**
 * @brief Handle to manage resources needed by cuML algorithms.
 */
class cumlHandle {
 public:
  /**
     * @brief construct a cumlHandle with default paramters.
     * @param n_streams number of internal streams to be setup
     *
     * The default paramters are 
     *   - stream: default or NULL stream
     *   - DeviceAllocator: cudaMalloc
     *   - HostAllocator: cudaMallocHost
     * @{
     */
  cumlHandle(int n_streams);
  cumlHandle();
  /** @} */
  /**
     * @brief releases all resources internally manged by cumlHandle.
     */
  ~cumlHandle();
  /**
     * @brief sets the stream to which all cuML work issued via this handle should be ordered.
     *
     * @param[in] stream    the stream to which cuML work should be ordered.
     */
  void setStream(cudaStream_t stream);
  /**
     * @brief gets the stream to which all cuML work issued via this handle should be ordered.
     *
     * @returns the stream to which cuML work should be ordered.
     */
  cudaStream_t getStream() const;
  /** Get the cached device properties of the device this handle is for */
  const cudaDeviceProp& getDeviceProperties() const;
  /**
     * @brief sets the allocator to use for all device allocations done in cuML.
     * 
     * @param[in] allocator     the MLCommon::deviceAllocator to use for device allocations.
     */
  void setDeviceAllocator(std::shared_ptr<deviceAllocator> allocator);
  /**
     * @brief gets the allocator to use for all device allocations done in cuML.
     * 
     * @returns the MLCommon::deviceAllocator to use for device allocations.
     */
  std::shared_ptr<deviceAllocator> getDeviceAllocator() const;
  /**
     * @brief sets the allocator to use for substantial host allocations done in cuML.
     * 
     * @param[in] allocator     the MLCommon::hostAllocator to use for host allocations.
     */
  void setHostAllocator(std::shared_ptr<hostAllocator> allocator);
  /**
     * @brief gets the allocator to use for substantial host allocations done in cuML.
     * 
     * @returns the MLCommon::hostAllocator to use for host allocations.
     */
  std::shared_ptr<hostAllocator> getHostAllocator() const;
  /**
     * @brief for internal use only.
     */
  const cumlHandle_impl& getImpl() const;
  /**
     * @brief for internal use only.
     */
  cumlHandle_impl& getImpl();

  /** for internal use only */
  static int getDefaultNumInternalStreams();

 private:
  //TODO: What is the right number?
  static constexpr int _default_num_internal_streams = 3;
  std::unique_ptr<cumlHandle_impl> _impl;
};

}  // end namespace ML
