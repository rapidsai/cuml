/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
