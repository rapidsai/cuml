/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/matrix/kernel_params.hpp>

#include <cuvs/distance/distance.hpp>

namespace ML::matrix {

inline cuvs::distance::kernels::KernelParams to_cuvs(KernelParams const& config)
{
  cuvs::distance::kernels::KernelParams params;

  // cuVS has no PRECOMPUTED kernel type. The kernel value is ignored by precomputed paths.
  if (config.kernel == KernelType::PRECOMPUTED) {
    params.kernel = cuvs::distance::kernels::KernelType::LINEAR;
  } else {
    params.kernel = static_cast<cuvs::distance::kernels::KernelType>(config.kernel);
  }
  params.degree = config.degree;
  params.gamma  = config.gamma;
  params.coef0  = config.coef0;

  return params;
}

}  // namespace ML::matrix
