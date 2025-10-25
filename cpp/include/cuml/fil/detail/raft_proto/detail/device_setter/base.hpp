/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>

namespace raft_proto {
namespace detail {

/** Struct for setting current device within a code block */
template <device_type D>
struct device_setter {
  device_setter(device_id<D> device) {}
};

}  // namespace detail
}  // namespace raft_proto
