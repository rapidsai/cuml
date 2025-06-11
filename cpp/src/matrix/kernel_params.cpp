/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuml/matrix/kernel_params.hpp>

#include <cuvs/distance/distance.hpp>

namespace ML::matrix {

cuvs::distance::kernels::KernelParams KernelParams::to_cuvs() const
{
  cuvs::distance::kernels::KernelParams params;

  params.kernel = static_cast<cuvs::distance::kernels::KernelType>(this->kernel);
  params.degree = this->degree;
  params.gamma  = this->gamma;
  params.coef0  = this->coef0;

  return params;
}

}  // end namespace ML::matrix
