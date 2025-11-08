/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

namespace raft_proto {
template <typename T, typename U>
HOST DEVICE auto constexpr ceildiv(T dividend, U divisor)
{
  return (dividend + divisor - T{1}) / divisor;
}
}  // namespace raft_proto
