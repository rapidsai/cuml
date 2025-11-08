/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/detail/host_only_throw/base.hpp>
#include <cuml/fil/detail/raft_proto/detail/host_only_throw/cpu.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

namespace raft_proto {
template <typename T, bool host = !GPU_COMPILATION>
using host_only_throw = detail::host_only_throw<T, host>;
}
