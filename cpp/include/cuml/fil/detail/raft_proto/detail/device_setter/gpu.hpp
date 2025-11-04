/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/cuda_check.hpp>
#include <cuml/fil/detail/raft_proto/detail/device_setter/base.hpp>
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime_api.h>

namespace raft_proto {
namespace detail {

/** Struct for setting current device within a code block */
template <>
struct device_setter<device_type::gpu> {
  device_setter(raft_proto::device_id<device_type::gpu> device) noexcept(false)
    : prev_device_{[]() {
        auto result = int{};
        raft_proto::cuda_check(cudaGetDevice(&result));
        return result;
      }()}
  {
    raft_proto::cuda_check(cudaSetDevice(device.value()));
  }

  ~device_setter() { RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_.value())); }

 private:
  device_id<device_type::gpu> prev_device_;
};

}  // namespace detail
}  // namespace raft_proto
