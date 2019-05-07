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

#include "cuda_utils.h"


namespace MLCommon {

/** functor for allocating a chunk of memory */
typedef void *(*AllocFunctor)(size_t);

/** functor for deallocating a chunk of memory */
typedef void (*DeallocFunctor)(void *, cudaStream_t);


/**
 * @brief An interface class to provide custom memory management interface
 *  to all APIs in ml-prims requiring temporary memory
 *
 * <pre>
 * DeviceAllocator mgr;
 * int* d_buff = (int*)mgr.alloc(nElems*sizeof(int));
 * mgr.free(d_buff);
 * </pre>
 */
class DeviceAllocator {
public:
  /**
   * @brief ctor. Creates an allocator object for custom memory management
   * using the functors for allocation and deallocation of device buffers.
   * @param a allocation functor
   * @param d deallocation functor
   */
  DeviceAllocator(AllocFunctor a, DeallocFunctor d) : allocF(a), deallocF(d) {}

  /** dtor */
  ~DeviceAllocator() {}

  /**
   * @brief function to allocate the requested chunk of data
   * @param size size in bytes of the buffer to be allocated
   * @return the allocator pointer
   */
  void *alloc(size_t size) { return allocF(size); }

  /**
   * @brief deallocate the input buffer
   * @param ptr the buffer to be freed
   * @param stream the cuda stream which will guarantee the safe free-up of
   *  this buffer. The internal memory manager can choose to use or ignore
   *  it, for the case of cudaFree which has an implicit such synchronize.
   */
  void free(void *ptr, cudaStream_t stream = 0) { deallocF(ptr, stream); }

private:
  /** functor for buffer allocation */
  AllocFunctor allocF;
  /** functor for buffer deallocation */
  DeallocFunctor deallocF;
};


/** Create a default allocator object which just uses cudaMalloc and cudaFree */
inline DeviceAllocator makeDefaultAllocator() {
  auto a = [](size_t nBytes) -> void * {
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, nBytes));
    return ptr;
  };
  auto d = [](void *ptr, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaFree(ptr));
  };
  return DeviceAllocator(a, d);
}

}; // end namespace MLCommon
