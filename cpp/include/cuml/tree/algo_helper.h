/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace ML {
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

};  // namespace ML
