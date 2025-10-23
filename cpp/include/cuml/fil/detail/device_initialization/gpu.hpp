/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_setter.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

#include <type_traits>

namespace ML {
namespace fil {
namespace detail {
namespace device_initialization {

/* Non-CUDA header declaration of the GPU specialization for device
 * initialization
 */
template <typename forest_t, raft_proto::device_type D>
std::enable_if_t<std::conjunction_v<std::bool_constant<raft_proto::GPU_ENABLED>,
                                    std::bool_constant<D == raft_proto::device_type::gpu>>,
                 void>
initialize_device(raft_proto::device_id<D> device);

}  // namespace device_initialization
}  // namespace detail
}  // namespace fil

}  // namespace ML
