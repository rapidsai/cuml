/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/fil/detail/raft_proto/detail/device_id/base.hpp>
#include <cuml/fil/detail/raft_proto/detail/device_id/cpu.hpp>
#ifdef CUML_ENABLE_GPU
#include <cuml/fil/detail/raft_proto/detail/device_id/gpu.hpp>
#endif
#include <cuml/fil/detail/raft_proto/device_type.hpp>

#include <variant>

namespace raft_proto {
template <device_type D>
using device_id = detail::device_id<D>;

using device_id_variant = std::variant<device_id<device_type::cpu>, device_id<device_type::gpu>>;
}  // namespace raft_proto
