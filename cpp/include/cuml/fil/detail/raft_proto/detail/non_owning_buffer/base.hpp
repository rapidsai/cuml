/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/device_type.hpp>

#include <memory>
#include <type_traits>

namespace raft_proto {
namespace detail {
template <device_type D, typename T>
struct non_owning_buffer {
  // TODO(wphicks): Assess need for buffers of const T
  using value_type = std::remove_const_t<T>;
  non_owning_buffer() : data_{nullptr} {}

  non_owning_buffer(T* ptr) : data_{ptr} {}

  auto* get() const { return data_; }

 private:
  // TODO(wphicks): Back this with RMM-allocated host memory
  T* data_;
};
}  // namespace detail
}  // namespace raft_proto
