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
#include <cuml/fil/detail/raft_proto/device_type.hpp>

#include <memory>
#include <type_traits>

namespace raft_proto {
namespace detail {
template <typename T>
struct owning_buffer<device_type::cpu, T> {
  // TODO(wphicks): Assess need for buffers of const T
  using value_type = std::remove_const_t<T>;

  owning_buffer() : data_{std::unique_ptr<T[]>{nullptr}} {}

  owning_buffer(std::size_t size) : data_{std::make_unique<T[]>(size)} {}

  auto* get() const { return data_.get(); }

 private:
  // TODO(wphicks): Back this with RMM-allocated host memory
  std::unique_ptr<T[]> data_;
};
}  // namespace detail
}  // namespace raft_proto
