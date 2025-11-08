/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/detail/owning_buffer/cpu.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#ifdef CUML_ENABLE_GPU
#include <cuml/fil/detail/raft_proto/detail/owning_buffer/gpu.hpp>
#endif
namespace raft_proto {
template <device_type D, typename T>
using owning_buffer = detail::owning_buffer<D, T>;
}
