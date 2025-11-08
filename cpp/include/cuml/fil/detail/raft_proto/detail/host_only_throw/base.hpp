/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

namespace raft_proto {
namespace detail {
template <typename T, bool host>
struct host_only_throw {
  template <typename... Args>
  host_only_throw(Args&&... args)
  {
    static_assert(host);  // Do not allow constexpr branch to compile if !host
  }
};
}  // namespace detail
}  // namespace raft_proto
