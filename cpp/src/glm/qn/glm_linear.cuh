/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/cuda_utils.cuh>
#include <raft/linalg/binary_op.cuh>
#include "glm_base.cuh"
#include "simple_mat.cuh"

namespace ML {
namespace GLM {

template <typename T>
struct SquaredLoss : GLMBase<T, SquaredLoss<T>> {
  typedef GLMBase<T, SquaredLoss<T>> Super;

  SquaredLoss(const raft::handle_t& handle, int D, bool has_bias) : Super(handle, D, 1, has_bias) {}

  inline __device__ T lz(const T y, const T z) const
  {
    T diff = y - z;
    return diff * diff * 0.5;
  }

  inline __device__ T dlz(const T y, const T z) const { return z - y; }
};

};  // namespace GLM
};  // namespace ML
