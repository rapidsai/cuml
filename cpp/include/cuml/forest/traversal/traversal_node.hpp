/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
