/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
