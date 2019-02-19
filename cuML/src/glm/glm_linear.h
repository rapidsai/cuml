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
#include <glm/glm_vectors.h>

namespace ML {
namespace GLM {

template <typename T, STORAGE_ORDER Storage = COL_MAJOR>
struct SquaredLoss1 : GLMBase<T, SquaredLoss1<T, Storage>, Storage> {
  typedef GLMBase<T, SquaredLoss1<T, Storage>, Storage> Super;

  SquaredLoss1(T *X, T *y, T *eta, int N, int D, bool has_bias, T lambda2)
      : Super(X, y, eta, N, D, has_bias, lambda2) {}

  inline __device__ T eval_l(const T y, const T eta) const {
    T diff = y - eta;
    return diff * diff * 0.5;
  }

  inline void eval_dl(const T *y, T *eta) {
    auto f = [] __device__(const T y, const T eta) { return (eta - y); };
    MLCommon::LinAlg::binaryOp(eta, y, eta, Super::N, f);
  }
};

}; // namespace GLM
}; // namespace ML
