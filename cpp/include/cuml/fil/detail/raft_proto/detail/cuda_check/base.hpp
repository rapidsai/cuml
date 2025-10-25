/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/device_type.hpp>

namespace raft_proto {
namespace detail {

template <device_type D, typename error_t>
void cuda_check(error_t const& err)
{
}

}  // namespace detail
}  // namespace raft_proto
