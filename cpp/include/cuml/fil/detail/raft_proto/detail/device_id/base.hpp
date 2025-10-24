/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/device_type.hpp>

namespace raft_proto {
namespace detail {
template <device_type D>
struct device_id {
  using value_type = int;

  device_id(value_type device_index) {}
  auto value() const { return value_type{}; }
};
}  // namespace detail
}  // namespace raft_proto
