/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
