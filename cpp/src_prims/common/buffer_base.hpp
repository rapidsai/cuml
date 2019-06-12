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

namespace MLCommon {

/**
 * @todo: Add missing doxygen documentation
 */
template <typename T>
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

  buffer_base(cudaStream_t stream, size_type n)
    : _size(n), _capacity(n), _data(nullptr), _stream(stream) {}

  ~buffer_base() {}

  value_type* data() { return _data; }

  const value_type* data() const { return _data; }

  size_type size() const { return _size; }

  void clear() { _size = 0; }

  iterator begin() { return _data; }

  const_iterator begin() const { return _data; }

  iterator end() { return _data + _size; }

  const_iterator end() const { return _data + _size; }

 protected:
  size_type _size;
  size_type _capacity;
  value_type* _data;
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
  cudaStream_t get_stream() const { return _stream; }

 private:
  cudaStream_t _stream;
};

}  // namespace MLCommon
