/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/detail/device_setter/base.hpp>
#ifdef CUML_ENABLE_GPU
#include <cuml/fil/detail/raft_proto/detail/device_setter/gpu.hpp>
#endif
#include <cuml/fil/detail/raft_proto/device_type.hpp>

namespace raft_proto {

using device_setter = detail::device_setter<device_type::gpu>;

}
