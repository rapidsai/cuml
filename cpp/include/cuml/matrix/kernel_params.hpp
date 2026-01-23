/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::distance::kernels {

struct KernelParams;

}  // end namespace cuvs::distance::kernels

namespace ML::matrix {

enum class KernelType { LINEAR, POLYNOMIAL, RBF, TANH, PRECOMPUTED };

struct KernelParams {
  KernelType kernel;
  int degree;
  double gamma;
  double coef0;

  /**
   * @brief Convert to cuvs KernelParams.
   *
   * @note For PRECOMPUTED kernels, the returned cuvs params will have kernel_type
   *       set to LINEAR as a placeholder, since cuvs doesn't have a PRECOMPUTED type.
   *       The kernel value won't be used in this case.
   */
  cuvs::distance::kernels::KernelParams to_cuvs() const;
};

}  // end namespace ML::matrix
