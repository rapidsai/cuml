/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuml/common/export.hpp>

namespace CUML_EXPORT ML {
enum CRITERION {
  GINI,
  ENTROPY,
  MSE,
  MAE,
  POISSON,
  GAMMA,
  INVERSE_GAUSSIAN,
  CRITERION_END,
};

};  // namespace CUML_EXPORT ML
