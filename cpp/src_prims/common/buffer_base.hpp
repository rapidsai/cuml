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

#include <cuda_runtime.h>
#include <memory>

#include "../utils.h"

namespace MLCommon {

/**
 * @todo: Add missing doxygen documentation
 */
template <typename T, typename Allocator>
class buffer_base {
 public:
  using size_type = std::size_t;
  using value_type = T;
  using iterator = value_type*;
  using const_iterator = const value_type*;
  using reference = T&;
  using const_reference = const T&;

  buffer_base() = delete;

  buffer_base(const buffer_base& other) = delete;

  buffer_base& operator=(const buffer_base& other) = delete;

  buffer_base(std::shared_ptr<Allocator> allocator, cudaStream_t stream,
              size_type n = 0)
    : _size(n),
      _capacity(n),
      _data(nullptr),
      _stream(stream),
      _allocator(allocator) {
    if (_capacity > 0) {
      _data = static_cast<value_type*>(
        _allocator->allocate(_capacity * sizeof(value_type), _stream));
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    }
  }

  ~buffer_base() {
    if (nullptr != _data) {
      _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
    }
  }

  value_type* data() { return _data; }

  const value_type* data() const { return _data; }

  size_type size() const { return _size; }

  void clear() { _size = 0; }

  iterator begin() { return _data; }

  const_iterator begin() const { return _data; }

  iterator end() { return _data + _size; }

  const_iterator end() const { return _data + _size; }

  void reserve(const size_type new_capacity, cudaStream_t stream) {
    set_stream(stream);
    if (new_capacity > _capacity) {
      value_type* new_data = static_cast<value_type*>(
        _allocator->allocate(new_capacity * sizeof(value_type), _stream));
      if (_size > 0) {
        CUDA_CHECK(cudaMemcpyAsync(new_data, _data, _size * sizeof(value_type),
                                   cudaMemcpyDefault, _stream));
      }
      if (nullptr != _data) {
        _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
      }
      _data = new_data;
      _capacity = new_capacity;
    }
  }

  void resize(const size_type new_size, cudaStream_t stream) {
    reserve(new_size, stream);
    _size = new_size;
  }

  void release(cudaStream_t stream) {
    set_stream(stream);
    if (nullptr != _data) {
      _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
    }
    _data = nullptr;
    _capacity = 0;
    _size = 0;
  }

  std::shared_ptr<Allocator> getAllocator() const { return _allocator; }

 protected:
  value_type* _data;

 private:
  size_type _size;
  size_type _capacity;
  cudaStream_t _stream;
  std::shared_ptr<Allocator> _allocator;

  void set_stream(cudaStream_t stream) {
    if (_stream != stream) {
      cudaEvent_t event;
      CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(event, _stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
      _stream = stream;
      CUDA_CHECK(cudaEventDestroy(event));
    }
  }
};

}  // namespace MLCommon
