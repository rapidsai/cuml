/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace ML {

// The type of season that represents the time-series
// ADDITIVE is with stable trend and level over periods
// MULTIPLICATIVE is with increasing/decreasing trend and level over periods
enum SeasonalType { ADDITIVE, MULTIPLICATIVE };

enum OptimCriterion {
  OPTIM_BFGS_ITER_LIMIT = 0,
  OPTIM_MIN_PARAM_DIFF  = 1,
  OPTIM_MIN_ERROR_DIFF  = 2,
  OPTIM_MIN_GRAD_NORM   = 3,
};

// These are the parameters used for optimizing alpha, beta, and gamma values
template <typename Dtype>
struct OptimParams {
  Dtype eps;
  Dtype min_param_diff;
  Dtype min_error_diff;
  Dtype min_grad_norm;
  int bfgs_iter_limit;
  int linesearch_iter_limit;
  Dtype linesearch_tau;
  Dtype linesearch_c;
  Dtype linesearch_step_size;
};

enum Norm { L0, L1, L2, LINF };

}  // namespace ML
