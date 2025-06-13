/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <cstddef>
#include <exception>
#include <string>

namespace ML {
namespace forest {

/** Exception indicating model is incompatible with FIL */
struct parentless_node_exception : std::exception {
  parentless_node_exception() : msg_{"Node does not track its parent"} {}
  parentless_node_exception(std::string msg) : msg_{msg} {}
  parentless_node_exception(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_.c_str(); }

 private:
  std::string msg_;
};

template <typename id_t = std::size_t>
struct traversal_node {
 public:
  using id_type                         = id_t;
  virtual bool is_leaf() const          = 0;
  virtual id_type hot_child() const     = 0;
  virtual id_type distant_child() const = 0;
  virtual id_type parent() const
  {
    throw parentless_node_exception();
    return id_type{};
  }
};

}  // namespace forest
}  // namespace ML
