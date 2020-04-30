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

#include <common/cudart_utils.h>
#include <cuda_utils.h>
#include <linalg/add.h>
#include <linalg/eltwise.h>
#include <linalg/norm.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include "sign.h"

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
  LinAlg::rowNorm(out, coef, len, 1, LinAlg::NormType::L1Norm, true, stream);
  LinAlg::scalarMultiply(out, out, alpha, 1, stream);
}

template <typename math_t>
void lassoGrad(math_t *grad, const math_t *coef, const int len,
               const math_t alpha, cudaStream_t stream) {
  sign(grad, coef, alpha, len, stream);
}

template <typename math_t>
void ridge(math_t *out, const math_t *coef, const int len, const math_t alpha,
           cudaStream_t stream) {
  LinAlg::rowNorm(out, coef, len, 1, LinAlg::NormType::L2Norm, true, stream);
  LinAlg::scalarMultiply(out, out, alpha, 1, stream);
}

template <typename math_t>
void ridgeGrad(math_t *grad, const math_t *coef, const int len,
               const math_t alpha, cudaStream_t stream) {
  LinAlg::scalarMultiply(grad, coef, math_t(2) * alpha, len, stream);
}

template <typename math_t>
void elasticnet(math_t *out, const math_t *coef, const int len,
                const math_t alpha, const math_t l1_ratio,
                cudaStream_t stream) {
  math_t *out_lasso = NULL;
  allocate(out_lasso, 1);

  ridge(out, coef, len, alpha * (math_t(1) - l1_ratio), stream);
  lasso(out_lasso, coef, len, alpha * l1_ratio, stream);

  LinAlg::add(out, out, out_lasso, 1, stream);

  if (out_lasso != NULL) {
    CUDA_CHECK(cudaFree(out_lasso));
  }
}

template <typename math_t>
void elasticnetGrad(math_t *grad, const math_t *coef, const int len,
                    const math_t alpha, const math_t l1_ratio,
                    cudaStream_t stream) {
  math_t *grad_lasso = NULL;
  allocate(grad_lasso, len);

  ridgeGrad(grad, coef, len, alpha * (math_t(1) - l1_ratio), stream);
  lassoGrad(grad_lasso, coef, len, alpha * l1_ratio, stream);

  LinAlg::add(grad, grad, grad_lasso, len, stream);

  if (grad_lasso != NULL) {
    CUDA_CHECK(cudaFree(grad_lasso));
  }
}

};  // namespace Functions
};  // namespace MLCommon
// end namespace ML
