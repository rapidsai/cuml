/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

#include <type_traits>

namespace ML {
namespace fil {
namespace detail {
namespace device_initialization {

/* Specialization for any initialization required for CPUs
 *
 * This specialization will also be used for non-GPU-enabled builds
 * (as a GPU no-op).
 */
template <typename forest_t, raft_proto::device_type D>
std::enable_if_t<std::disjunction_v<std::bool_constant<!raft_proto::GPU_ENABLED>,
                                    std::bool_constant<D == raft_proto::device_type::cpu>>,
                 void>
initialize_device(raft_proto::device_id<D> device)
{
}

}  // namespace device_initialization
}  // namespace detail
}  // namespace fil
}  // namespace ML
