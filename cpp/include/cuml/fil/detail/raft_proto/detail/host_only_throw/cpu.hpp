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
#include <cuml/experimental/fil/detail/raft_proto/detail/host_only_throw/base.hpp>
#include <cuml/experimental/fil/detail/raft_proto/gpu_support.hpp>

namespace raft_proto {
namespace detail {
template <typename T>
struct host_only_throw<T, true> {
  template <typename... Args>
  host_only_throw(Args&&... args) noexcept(false)
  {
    throw T{std::forward<Args>(args)...};
  }
};
}  // namespace detail
}  // namespace raft_proto
