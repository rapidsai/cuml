/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "cuda_utils.h"
#include "glm/glm_base.h"
#include "linalg/binary_op.h"
#include <matrix/simple_mat.h>

namespace ML {
namespace GLM {

template <typename T>
struct SquaredLoss : GLMBase<T, SquaredLoss<T>> {
  typedef GLMBase<T, SquaredLoss<T>> Super;

  SquaredLoss(int D, bool has_bias, const cublasHandle_t & cublas)
      : Super(D, 1, has_bias, cublas) {}

  inline __device__ T lz(const T y, const T z) const {
    T diff = y - z;
    return diff * diff * 0.5;
  }

  inline __device__ T dlz(const T y, const T z) const {
    return z - y;
  }

};

}; // namespace GLM
}; // namespace ML
