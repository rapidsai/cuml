/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

  cuvs::distance::kernels::KernelParams to_cuvs() const;
};

}  // end namespace ML::matrix
