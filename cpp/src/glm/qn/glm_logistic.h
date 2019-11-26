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

#include <glm/qn/simple_mat.h>
#include "cuda_utils.h"
#include "glm/qn/glm_base.h"
#include "linalg/binary_op.h"

namespace ML {
namespace GLM {

template <typename T>
struct LogisticLoss : GLMBase<T, LogisticLoss<T>> {
  typedef GLMBase<T, LogisticLoss<T>> Super;

  LogisticLoss(const cumlHandle_impl &handle, int D, bool has_bias)
    : Super(handle, D, 1, has_bias) {}

  inline __device__ T log_sigmoid(T x) const {
    return -MLCommon::myLog(1 + MLCommon::myExp(-x));
  }

  inline __device__ T lz(const T y, const T z) const {
    T ytil = 2 * y - 1;
    return -log_sigmoid(ytil * z);
  }

  inline __device__ T dlz(const T y, const T z) const {
    return T(1.0) / (T(1.0) + MLCommon::myExp(-z)) - y;
  }
};
};  // namespace GLM
};  // namespace ML
