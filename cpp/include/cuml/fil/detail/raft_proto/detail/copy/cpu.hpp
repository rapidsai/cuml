/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

#include <stdint.h>

#include <algorithm>
#include <cstring>

namespace raft_proto {
namespace detail {

template <device_type dst_type, device_type src_type, typename T>
std::enable_if_t<std::conjunction_v<std::bool_constant<dst_type == device_type::cpu>,
                                    std::bool_constant<src_type == device_type::cpu>>,
                 void>
copy(T* dst, T const* src, uint32_t size, cuda_stream stream)
{
  std::copy(src, src + size, dst);
}

template <device_type dst_type, device_type src_type, typename T>
std::enable_if_t<
  std::conjunction_v<std::disjunction<std::bool_constant<dst_type != device_type::cpu>,
                                      std::bool_constant<src_type != device_type::cpu>>,
                     std::bool_constant<!GPU_ENABLED>>,
  void>
copy(T* dst, T const* src, uint32_t size, cuda_stream stream)
{
  throw gpu_unsupported("Copying from or to device in non-GPU build");
}

}  // namespace detail
}  // namespace raft_proto
