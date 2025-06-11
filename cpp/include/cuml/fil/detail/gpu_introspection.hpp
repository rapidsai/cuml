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
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/raft_proto/cuda_check.hpp>
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>

#include <cuda_runtime_api.h>

#include <vector>

namespace ML {
namespace fil {
namespace detail {

inline auto get_max_shared_mem_per_block(
  raft_proto::device_id<raft_proto::device_type::gpu> device_id)
{
  auto thread_local cache = std::vector<int>{};
  if (cache.size() == 0) {
    auto device_count = int{};
    raft_proto::cuda_check(cudaGetDeviceCount(&device_count));
    cache.resize(device_count);
    for (auto dev = 0; dev < device_count; ++dev) {
      raft_proto::cuda_check(
        cudaDeviceGetAttribute(&(cache[dev]), cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    }
  }
  return index_type(cache.at(device_id.value()));
}

inline auto get_sm_count(raft_proto::device_id<raft_proto::device_type::gpu> device_id)
{
  auto thread_local cache = std::vector<int>{};
  if (cache.size() == 0) {
    auto device_count = int{};
    raft_proto::cuda_check(cudaGetDeviceCount(&device_count));
    cache.resize(device_count);
    for (auto dev = 0; dev < device_count; ++dev) {
      raft_proto::cuda_check(
        cudaDeviceGetAttribute(&(cache[dev]), cudaDevAttrMultiProcessorCount, dev));
    }
  }
  return index_type(cache.at(device_id.value()));
}

inline auto get_max_threads_per_sm(raft_proto::device_id<raft_proto::device_type::gpu> device_id)
{
  auto result = int{};
  raft_proto::cuda_check(
    cudaDeviceGetAttribute(&result, cudaDevAttrMaxThreadsPerMultiProcessor, device_id.value()));
  return index_type(result);
}

inline auto get_max_shared_mem_per_sm(raft_proto::device_id<raft_proto::device_type::gpu> device_id)
{
  auto thread_local cache = std::vector<int>{};
  if (cache.size() == 0) {
    auto device_count = int{};
    raft_proto::cuda_check(cudaGetDeviceCount(&device_count));
    cache.resize(device_count);
    for (auto dev = 0; dev < device_count; ++dev) {
      raft_proto::cuda_check(
        cudaDeviceGetAttribute(&(cache[dev]), cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    }
  }
  return index_type(cache.at(device_id.value()));
}

inline auto get_mem_clock_rate(raft_proto::device_id<raft_proto::device_type::gpu> device_id)
{
  auto result = int{};
  raft_proto::cuda_check(
    cudaDeviceGetAttribute(&result, cudaDevAttrMemoryClockRate, device_id.value()));
  return index_type(result);
}

inline auto get_core_clock_rate(raft_proto::device_id<raft_proto::device_type::gpu> device_id)
{
  auto result = int{};
  raft_proto::cuda_check(cudaDeviceGetAttribute(&result, cudaDevAttrClockRate, device_id.value()));
  return index_type(result);
}

/* The maximum number of bytes that can be read in a single instruction */
auto constexpr static const MAX_READ_CHUNK        = index_type{128};
auto constexpr static const MAX_BLOCKS            = index_type{65536};
auto constexpr static const WARP_SIZE             = index_type{32};
auto constexpr static const MAX_THREADS_PER_BLOCK = index_type{256};
#ifdef __CUDACC__
#if __CUDA_ARCH__ == 720 || __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860 || \
  __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
auto constexpr static const MAX_THREADS_PER_SM = index_type{1024};
#else
auto constexpr static const MAX_THREADS_PER_SM = index_type{2048};
#endif
#else
auto constexpr static const MAX_THREADS_PER_SM = index_type{2048};
#endif

auto constexpr static const MIN_BLOCKS_PER_SM = MAX_THREADS_PER_SM / MAX_THREADS_PER_BLOCK;

}  // namespace detail
}  // namespace fil
}  // namespace ML
