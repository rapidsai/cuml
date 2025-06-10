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
#include <cuml/fil/detail/raft_proto/detail/owning_buffer/base.hpp>
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_setter.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <type_traits>

namespace raft_proto {
namespace detail {
template <typename T>
struct owning_buffer<device_type::gpu, T> {
  // TODO(wphicks): Assess need for buffers of const T
  using value_type = std::remove_const_t<T>;
  owning_buffer() : data_{} {}

  owning_buffer(device_id<device_type::gpu> device_id,
                std::size_t size,
                cudaStream_t stream) noexcept(false)
    : data_{[&device_id, &size, &stream]() {
        auto device_context = device_setter{device_id};
        return rmm::device_buffer{size * sizeof(value_type), rmm::cuda_stream_view{stream}};
      }()}
  {
  }

  auto* get() const { return reinterpret_cast<T*>(data_.data()); }

 private:
  mutable rmm::device_buffer data_;
};
}  // namespace detail
}  // namespace raft_proto
