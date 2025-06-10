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
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>

#include <algorithm>
#include <cstddef>
#ifdef CUML_ENABLE_GPU
#include <raft/core/handle.hpp>
#endif

namespace raft_proto {
#ifdef CUML_ENABLE_GPU
struct handle_t {
  handle_t(raft::handle_t const* handle_ptr = nullptr) : raft_handle_{handle_ptr} {}
  handle_t(raft::handle_t const& raft_handle) : raft_handle_{&raft_handle} {}
  auto get_next_usable_stream() const
  {
    return raft_proto::cuda_stream{raft_handle_->get_next_usable_stream().value()};
  }
  auto get_stream_pool_size() const { return raft_handle_->get_stream_pool_size(); }
  auto get_usable_stream_count() const { return std::max(get_stream_pool_size(), std::size_t{1}); }
  void synchronize() const
  {
    raft_handle_->sync_stream_pool();
    raft_handle_->sync_stream();
  }

 private:
  // Have to store a pointer because handle is not movable
  raft::handle_t const* raft_handle_;
};
#else
struct handle_t {
  auto get_next_usable_stream() const { return raft_proto::cuda_stream{}; }
  auto get_stream_pool_size() const { return std::size_t{}; }
  auto get_usable_stream_count() const { return std::max(get_stream_pool_size(), std::size_t{1}); }
  void synchronize() const {}
};
#endif
}  // namespace raft_proto
