/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

namespace ML {

template <typename T>
class pinned_host_vector {
 public:
  pinned_host_vector() = default;

  explicit pinned_host_vector(std::size_t n)
    : size_{n}, data_{static_cast<T*>(pinned_mr.allocate(rmm::cuda_stream_default, n * sizeof(T)))}
  {
    std::uninitialized_fill(data_, data_ + n, static_cast<T>(0));
  }
  ~pinned_host_vector()
  {
    pinned_mr.deallocate(rmm::cuda_stream_default, data_, size_ * sizeof(T));
  }

  pinned_host_vector(pinned_host_vector const&)            = delete;
  pinned_host_vector(pinned_host_vector&&)                 = delete;
  pinned_host_vector& operator=(pinned_host_vector const&) = delete;
  pinned_host_vector& operator=(pinned_host_vector&&)      = delete;

  void resize(std::size_t n)
  {
    size_ = n;
    data_ = static_cast<T*>(pinned_mr.allocate(rmm::cuda_stream_default, n * sizeof(T)));
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
