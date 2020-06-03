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

#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/cuml.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>

namespace ML {

/**
 * @brief Implemententation of ML::deviceAllocator using the
 *        RAPIDS Memory Manager (RMM) for allocations.
 *
 * rmmAllocatorAdapter does not initialize RMM. If RMM is not initialized on
 * construction of rmmAllocatorAdapter allocations fall back to cudaMalloc.
 */
class rmmAllocatorAdapter : public ML::deviceAllocator {
 public:
  rmmAllocatorAdapter() {}

  /**
   * @brief asynchronosly allocate n bytes that can be used after all work in
   *        stream sheduled prior to this call has completetd.
   *
   * @param[in] n         size of the allocation in bytes
   * @param[in] stream    the stream to use for the asynchronous allocations
   */
  virtual void* allocate(std::size_t n, cudaStream_t stream) {
    void* ptr = 0;
    ptr = rmm::mr::get_default_resource()->allocate(n, stream);
    return ptr;
  }

  /**
   * @brief asynchronosly free an allocation of n bytes that can be reused after
   *        all work in stream scheduled prior to this call has completed.
   *
   * @param[in] p         pointer to n bytes of memory to be deallocated
   * @param[in] n         size of the allocation to release in bytes
   * @param[in] stream    the stream to use for the asynchronous free
   */
  virtual void deallocate(void* p, std::size_t n, cudaStream_t stream) {
    rmm::mr::get_default_resource()->deallocate(p, n, stream);
  }

  virtual ~rmmAllocatorAdapter() {}
};

}  // end namespace ML
