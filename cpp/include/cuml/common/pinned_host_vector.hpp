/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <rmm/mr/pinned_host_memory_resource.hpp>

namespace ML {

template <typename T>
class pinned_host_vector {
 public:
  pinned_host_vector() = default;

  explicit pinned_host_vector(std::size_t n)
    : size_{n}, data_{static_cast<T*>(pinned_mr.allocate(n * sizeof(T)))}
  {
    std::uninitialized_fill(data_, data_ + n, static_cast<T>(0));
  }
  ~pinned_host_vector() { pinned_mr.deallocate(data_, size_ * sizeof(T)); }

  pinned_host_vector(pinned_host_vector const&)            = delete;
  pinned_host_vector(pinned_host_vector&&)                 = delete;
  pinned_host_vector& operator=(pinned_host_vector const&) = delete;
  pinned_host_vector& operator=(pinned_host_vector&&)      = delete;

  void resize(std::size_t n)
  {
    size_ = n;
    data_ = static_cast<T*>(pinned_mr.allocate(n * sizeof(T)));
    std::uninitialized_fill(data_, data_ + n, static_cast<T>(0));
  }

  T* data() { return data_; }

  T* begin() { return data_; }

  T* end() { return data_ + size_; }

  std::size_t size() { return size_; }

  T operator[](std::size_t idx) const { return *(data_ + idx); }
  T& operator[](std::size_t idx) { return *(data_ + idx); }

 private:
  rmm::mr::pinned_host_memory_resource pinned_mr{};
  T* data_;
  std::size_t size_;
};

}  // namespace ML
