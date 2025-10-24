/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <exception>
#include <string>

namespace ML {
namespace forest {
struct traversal_exception : std::exception {
  traversal_exception() : msg_{"Error encountered while traversing forest"} {}
  traversal_exception(std::string msg) : msg_{msg} {}
  traversal_exception(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_.c_str(); }

 private:
  std::string msg_;
};
}  // namespace forest
}  // namespace ML
