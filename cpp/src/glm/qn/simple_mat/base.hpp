/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/handle.hpp>
#include <raft/core/interruptible.hpp>
#include <raft/util/cuda_utils.cuh>

namespace ML {

template <typename T>
struct SimpleDenseMat;

template <typename T>
struct SimpleMat {
  int m, n;

  SimpleMat(int m, int n) : m(m), n(n) {}

  void operator=(const SimpleMat<T>& other) = delete;

  virtual void print(std::ostream& oss) const = 0;

  /**
   * GEMM assigning to C where `this` refers to B.
   *
   * ```
   * C <- alpha * A^transA * (*this)^transB + beta * C
   * ```
   */
  virtual void gemmb(const raft::handle_t& handle,
                     const T alpha,
                     const SimpleDenseMat<T>& A,
                     const bool transA,
                     const bool transB,
                     const T beta,
                     SimpleDenseMat<T>& C,
                     cudaStream_t stream) const = 0;
};

};  // namespace ML
