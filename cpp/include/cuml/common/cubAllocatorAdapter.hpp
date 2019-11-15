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

#include <cub/util_allocator.cuh>
#include <cuml/cuml.hpp>

namespace ML {

class cachingDeviceAllocator : public deviceAllocator {
 public:
  cachingDeviceAllocator()
    : _allocator(8, 3, cub::CachingDeviceAllocator::INVALID_BIN,
                 cub::CachingDeviceAllocator::INVALID_SIZE) {}

  void* allocate(std::size_t n, cudaStream_t stream) {
    void* ptr = 0;
    _allocator.DeviceAllocate(&ptr, n, stream);
    return ptr;
  }

  void deallocate(void* p, std::size_t, cudaStream_t) {
    _allocator.DeviceFree(p);
  }

 private:
  cub::CachingDeviceAllocator _allocator;
};

}  // end namespace ML
