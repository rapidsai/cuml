/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>

#include <type_traits>

namespace raft_proto {
namespace detail {

template <device_type D, typename T>
struct owning_buffer {
  owning_buffer() {}
  owning_buffer(device_id<D> device_id, std::size_t size, cuda_stream stream) {}
  auto* get() const { return static_cast<T*>(nullptr); }
};

}  // namespace detail
}  // namespace raft_proto
