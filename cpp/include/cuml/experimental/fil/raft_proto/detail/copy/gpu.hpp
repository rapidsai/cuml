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
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <cuml/experimental/raft_proto/cuda_check.hpp>
#include <cuml/experimental/raft_proto/cuda_stream.hpp>
#include <cuml/experimental/raft_proto/gpu_support.hpp>
#include <type_traits>

namespace raft_proto {
namespace detail {

template<device_type dst_type, device_type src_type, typename T>
std::enable_if_t<(dst_type == device_type::gpu || src_type == device_type::gpu) && GPU_ENABLED, void> copy(T* dst, T const* src, uint32_t size, cuda_stream stream) {
  raft_proto::cuda_check(cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDefault, stream));
}

}
}
