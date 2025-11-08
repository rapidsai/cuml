/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/detail/host_only_throw/base.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

namespace raft_proto {
namespace detail {
template <typename T>
struct host_only_throw<T, true> {
  template <typename... Args>
  host_only_throw(Args&&... args) noexcept(false)
  {
    throw T{std::forward<Args>(args)...};
  }
};
}  // namespace detail
}  // namespace raft_proto
