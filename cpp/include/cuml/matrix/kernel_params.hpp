/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/export.hpp>

namespace CUML_EXPORT ML {
namespace matrix {

enum class KernelType { LINEAR, POLYNOMIAL, RBF, TANH, PRECOMPUTED };

struct KernelParams {
  KernelType kernel;
  int degree;
  double gamma;
  double coef0;
};

}  // end namespace matrix
}  // end namespace CUML_EXPORT ML
