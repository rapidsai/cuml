/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

namespace cuvs::distance::kernels {

struct KernelParams;

}  // end namespace cuvs::distance::kernels

namespace ML::matrix {

enum class KernelType { LINEAR, POLYNOMIAL, RBF, TANH };

struct KernelParams {
  KernelType kernel;
  int degree;
  double gamma;
  double coef0;

  cuvs::distance::kernels::KernelParams to_cuvs() const;
};

}  // end namespace ML::matrix
