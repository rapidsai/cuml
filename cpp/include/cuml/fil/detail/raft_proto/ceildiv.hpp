/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

#include <type_traits>

namespace raft_proto {
template <typename T, typename U>
HOST DEVICE auto constexpr ceildiv(T dividend, U divisor)
{
  static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integers");
  return dividend / divisor + (dividend % divisor != 0);
}
}  // namespace raft_proto
