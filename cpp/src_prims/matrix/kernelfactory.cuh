/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "grammatrix.cuh"
#include "kernelmatrices.cuh"
#include <cuml/matrix/kernelparams.h>
#include <raft/cudart_utils.h>

namespace MLCommon {
namespace Matrix {

template <typename math_t>
class KernelFactory {
 public:
  static GramMatrixBase<math_t>* create(KernelParams params, cublasHandle_t cublas_handle)
  {
    GramMatrixBase<math_t>* res;
    // KernelParams is not templated, we convert the parameters to math_t here:
    math_t coef0 = params.coef0;
    math_t gamma = params.gamma;
    switch (params.kernel) {
      case LINEAR: res = new GramMatrixBase<math_t>(cublas_handle); break;
      case POLYNOMIAL:
        res = new PolynomialKernel<math_t, int>(params.degree, gamma, coef0, cublas_handle);
        break;
      case TANH: res = new TanhKernel<math_t>(gamma, coef0, cublas_handle); break;
      case RBF: res = new RBFKernel<math_t>(gamma); break;
      default: throw raft::exception("Kernel not implemented");
    }
    return res;
  }
};

};  // end namespace Matrix
};  // end namespace MLCommon
