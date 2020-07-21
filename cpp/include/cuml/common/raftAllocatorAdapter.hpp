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

#include "cuml_allocator.hpp"

#include <raft/mr/device/allocator.hpp>

namespace ML {

class raftDeviceAllocatorAdapter : public ML::deviceAllocator {
 public:

  raftDeviceAllocatorAdapter(std::shared_ptr<raft::mr::device::allocator> raftAllocator) : _raftAllocator(raftAllocator) {}
  raftDeviceAllocatorAdapter() {
    _raftAllocator = std::make_unique<raft::mr::device::default_allocator>();
  }

  virtual void* allocate(std::size_t n, cudaStream_t stream) {
    return _raftAllocator::allocate(n, stream);
  }

  virtual void deallocate(void* p, std::size_t n, cudaStream_t stream) {
    _raftAllocator::deallocate(p, n, stream);
  }

  std::shared_ptr<raft::mr::device::allocator> getRaftDeviceAllocator() {
    return _raftAllocator;
  }

  virtual ~raftDeviceAllocatorAdapter() {}

  private:
    std::shared_ptr<raft::mr::device::allocator> _raftAllocator;
};

}  // end namespace ML
