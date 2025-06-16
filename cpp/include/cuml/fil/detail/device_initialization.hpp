/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
