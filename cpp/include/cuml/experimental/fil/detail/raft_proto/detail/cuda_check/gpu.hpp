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
#include <cuml/experimental/fil/detail/raft_proto/detail/cuda_check/base.hpp>
#include <cuml/experimental/fil/detail/raft_proto/device_type.hpp>
#include <cuml/experimental/fil/detail/raft_proto/exceptions.hpp>
namespace raft_proto {
namespace detail {

template <>
inline void cuda_check<device_type::gpu, cudaError_t>(cudaError_t const& err) noexcept(false)
{
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw bad_cuda_call(cudaGetErrorString(err));
  }
}

}  // namespace detail
}  // namespace raft_proto
