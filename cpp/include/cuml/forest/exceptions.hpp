/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/common/export.hpp>

#include <exception>
#include <string>

namespace CUML_EXPORT ML {
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
}  // namespace CUML_EXPORT ML
