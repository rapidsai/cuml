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
