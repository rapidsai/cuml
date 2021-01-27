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

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <rmm/device_uvector.hpp>
#include "sign.cuh"

namespace MLCommon {
namespace Functions {

enum penalty {
  NONE,
  L1,
  L2,
  ELASTICNET,
};

template <typename math_t>
void lasso(math_t *out, const math_t *coef, const int len, const math_t alpha,
           cudaStream_t stream) {
  raft::linalg::rowNorm(out, coef, len, 1, raft::linalg::NormType::L1Norm, true,
                        stream);
  raft::linalg::scalarMultiply(out, out, alpha, 1, stream);
}

template <typename math_t>
void lassoGrad(math_t *grad, const math_t *coef, const int len,
               const math_t alpha, cudaStream_t stream) {
  sign(grad, coef, alpha, len, stream);
}

template <typename math_t>
void ridge(math_t *out, const math_t *coef, const int len, const math_t alpha,
           cudaStream_t stream) {
  raft::linalg::rowNorm(out, coef, len, 1, raft::linalg::NormType::L2Norm, true,
                        stream);
  raft::linalg::scalarMultiply(out, out, alpha, 1, stream);
}

template <typename math_t>
void ridgeGrad(math_t *grad, const math_t *coef, const int len,
               const math_t alpha, cudaStream_t stream) {
  raft::linalg::scalarMultiply(grad, coef, math_t(2) * alpha, len, stream);
}

template <typename math_t>
void elasticnet(math_t *out, const math_t *coef, const int len,
                const math_t alpha, const math_t l1_ratio,
                cudaStream_t stream) {
  rmm::device_uvector<math_t> out_lasso(1, stream);

  ridge(out, coef, len, alpha * (math_t(1) - l1_ratio), stream);
  lasso(out_lasso.data(), coef, len, alpha * l1_ratio, stream);

  raft::linalg::add(out, out, out_lasso.data(), 1, stream);
}

template <typename math_t>
void elasticnetGrad(math_t *grad, const math_t *coef, const int len,
                    const math_t alpha, const math_t l1_ratio,
                    cudaStream_t stream) {
  rmm::device_uvector<math_t> grad_lasso(len, stream);

  ridgeGrad(grad, coef, len, alpha * (math_t(1) - l1_ratio), stream);
  lassoGrad(grad_lasso.data(), coef, len, alpha * l1_ratio, stream);

  raft::linalg::add(grad, grad, grad_lasso.data(), len, stream);
}

};  // namespace Functions
};  // namespace MLCommon
// end namespace ML
