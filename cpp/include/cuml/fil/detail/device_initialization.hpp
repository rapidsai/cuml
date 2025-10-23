/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/fil/detail/device_initialization/cpu.hpp>

#include <variant>
#ifdef CUML_ENABLE_GPU
#include <cuml/fil/detail/device_initialization/gpu.hpp>
#endif

namespace ML {
namespace fil {
namespace detail {
/* Set any required device options for optimizing FIL compute */
template <typename forest_t, raft_proto::device_type D>
void initialize_device(raft_proto::device_id<D> device)
{
  device_initialization::initialize_device<forest_t>(device);
}

/* Set any required device options for optimizing FIL compute */
template <typename forest_t>
void initialize_device(raft_proto::device_id_variant device)
{
  std::visit(
    [](auto&& concrete_device) {
      device_initialization::initialize_device<forest_t>(concrete_device);
    },
    device);
}
}  // namespace detail
}  // namespace fil
}  // namespace ML
