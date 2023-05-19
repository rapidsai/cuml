/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuml/experimental/fil/detail/raft_proto/detail/device_id/base.hpp>
#include <cuml/experimental/fil/detail/raft_proto/detail/device_id/cpu.hpp>
#ifdef CUML_ENABLE_GPU
#include <cuml/experimental/fil/detail/raft_proto/detail/device_id/gpu.hpp>
#endif
#include <cuml/experimental/fil/detail/raft_proto/device_type.hpp>
#include <variant>

namespace raft_proto {
template <device_type D>
using device_id = detail::device_id<D>;

using device_id_variant = std::variant<device_id<device_type::cpu>, device_id<device_type::gpu>>;
}  // namespace raft_proto
