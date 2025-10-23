/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/common/logger.hpp>

namespace ML {
namespace SVM {

enum SvmType { C_SVC, NU_SVC, EPSILON_SVR, NU_SVR };

/**
 * Numerical input parameters for an SVM.
 *
 * There are several parameters that control how long we train. The training
 * stops if:
 * - max_iter iterations are reached. If you pass -1, then
 *   max_diff = 100 * n_rows
 * - the diff becomes less the tol
 * - the diff is changing less then 0.001*tol in nochange_steps consecutive
 *   outer iterations.
 */
struct SvmParameter {
  double C;           //!< Penalty term C
  double cache_size;  //!< kernel cache size in MiB
  //! maximum number of outer SMO iterations. Use -1 to let the SMO solver set
  //! a default value (100*n_rows).
  int max_iter;
  int nochange_steps;                   //<! Number of steps to continue with non-changing diff
  double tol;                           //!< Tolerance used to stop fitting.
  rapids_logger::level_enum verbosity;  //!< Print information about training
  double epsilon;                       //!< epsilon parameter for epsilon-SVR
  SvmType svmType;
};

};  // namespace SVM
};  // namespace ML
