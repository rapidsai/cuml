/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <exception>

namespace raft_proto {
struct bad_cuda_call : std::exception {
  bad_cuda_call() : bad_cuda_call("CUDA API call failed") {}
  bad_cuda_call(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

struct out_of_bounds : std::exception {
  out_of_bounds() : out_of_bounds("Attempted out-of-bounds memory access") {}
  out_of_bounds(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

struct wrong_device_type : std::exception {
  wrong_device_type() : wrong_device_type("Attempted to use host data on GPU or device data on CPU")
  {
  }
  wrong_device_type(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

struct mem_type_mismatch : std::exception {
  mem_type_mismatch() : mem_type_mismatch("Memory type does not match expected type") {}
  mem_type_mismatch(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

struct wrong_device : std::exception {
  wrong_device() : wrong_device("Attempted to use incorrect device") {}
  wrong_device(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

}  // namespace raft_proto
