/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
struct LogisticLoss : GLMBase<T, LogisticLoss<T>> {
  typedef GLMBase<T, LogisticLoss<T>> Super;

  LogisticLoss(const raft::handle_t &handle, int D, bool has_bias)
    : Super(handle, D, 1, has_bias) {}

  inline __device__ T log_sigmoid(T x) const {
    // To avoid floating point overflow in the exp function
    T temp = raft::myLog(1 + raft::myExp(x < 0 ? x : -x));
    return x < 0 ? x - temp : -temp;
  }

  inline __device__ T lz(const T y, const T z) const {
    T ytil = 2 * y - 1;
    return -log_sigmoid(ytil * z);
  }

  inline __device__ T dlz(const T y, const T z) const {
    // To avoid fp overflow with exp(z) when abs(z) is large
    T ez = raft::myExp(z < 0 ? z : -z);
    T numerator = z < 0 ? ez : T(1.0);
    return numerator / (T(1.0) + ez) - y;
  }
};
};  // namespace GLM
};  // namespace ML
