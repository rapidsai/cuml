/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/detail/device_id/base.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>

namespace raft_proto {
namespace detail {
template <>
struct device_id<device_type::cpu> {
  using value_type = int;
  device_id() : id_{value_type{}} {};
  device_id(value_type dev_id) : id_{dev_id} {};

  auto value() const noexcept { return id_; }

 private:
  value_type id_;
};
}  // namespace detail
}  // namespace raft_proto
