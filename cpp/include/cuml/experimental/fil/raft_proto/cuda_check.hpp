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
#include <cuml/experimental/raft_proto/detail/cuda_check/base.hpp>
#ifdef ENABLE_GPU
#include <cuml/experimental/raft_proto/detail/cuda_check/gpu.hpp>
#endif
#include <cuml/experimental/raft_proto/device_type.hpp>
#include <cuml/experimental/raft_proto/gpu_support.hpp>

namespace raft_proto {
template <typename error_t>
void cuda_check(error_t const& err) noexcept(!GPU_ENABLED) {
  detail::cuda_check<device_type::gpu>(err);
}
}
