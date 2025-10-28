/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/detail/non_owning_buffer/base.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>

namespace raft_proto {
template <device_type D, typename T>
using non_owning_buffer = detail::non_owning_buffer<D, T>;
}
