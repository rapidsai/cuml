/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/cuda_check.hpp>
#include <cuml/fil/detail/raft_proto/detail/device_id/base.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>

#include <rmm/cuda_device.hpp>

namespace raft_proto {
namespace detail {
template <>
struct device_id<device_type::gpu> {
  using value_type = typename rmm::cuda_device_id::value_type;
  device_id() noexcept(false)
    : id_{[]() {
        auto raw_id = value_type{};
        raft_proto::cuda_check(cudaGetDevice(&raw_id));
        return raw_id;
      }()} {};
  device_id(value_type dev_id) noexcept : id_{dev_id} {};

  auto value() const noexcept { return id_.value(); }

 private:
  rmm::cuda_device_id id_;
};
}  // namespace detail
}  // namespace raft_proto
