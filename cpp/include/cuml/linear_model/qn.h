/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
namespace ML::GLM {

extern "C" {
#endif

/** Loss function types supported by the Quasi-Newton solvers. */
enum qn_loss_type {
  /** Logistic classification.
   *  Expected target: {0, 1}.
   */
  QN_LOSS_LOGISTIC = 0,
  /** L2 regression.
   *  Expected target: R.
   */
  QN_LOSS_SQUARED = 1,
  /** Softmax classification..
   *  Expected target: {0, 1, ...}.
   */
  QN_LOSS_SOFTMAX = 2,
  /** Hinge.
   *  Expected target: {0, 1}.
   */
  QN_LOSS_SVC_L1 = 3,
  /** Squared-hinge.
   *  Expected target: {0, 1}.
   */
  QN_LOSS_SVC_L2 = 4,
  /** Epsilon-insensitive.
   *  Expected target: R.
   */
  QN_LOSS_SVR_L1 = 5,
  /** Epsilon-insensitive-squared.
   *  Expected target: R.
   */
  QN_LOSS_SVR_L2 = 6,
  /** L1 regression.
   *  Expected target: R.
   */
  QN_LOSS_ABS = 7,
  /** Someone forgot to set the loss type! */
  QN_LOSS_UNKNOWN = 99
};
#ifndef __cplusplus
typedef enum qn_loss_type qn_loss_type;
#endif

struct qn_params {
  /** Loss type. */
  qn_loss_type loss;
  /** Regularization: L1 component. */
  double penalty_l1;
  /** Regularization: L2 component. */
  double penalty_l2;
  /** Convergence criteria: the threshold on the gradient. */
  double grad_tol;
  /** Convergence criteria: the threshold on the function change. */
  double change_tol;
  /** Maximum number of iterations. */
  int max_iter;
  /** Maximum number of linesearch (inner loop) iterations. */
  int linesearch_max_iter;
  /** Number of vectors approximating the hessian (l-bfgs). */
  int lbfgs_memory;
  /** Triggers extra output when greater than zero. */
  int verbose;
  /** Whether to fit the bias term. */
  bool fit_intercept;
  /**
   * Whether to divide the L1 and L2 regularization parameters by the sample size.
   *
   * Note, the defined QN loss functions normally are scaled for the sample size,
   * e.g. the average across the data rows is calculated.
   * Enabling `penalty_normalized` makes this solver's behavior compatible to those solvers,
   * which do not scale the loss functions (like sklearn.LogisticRegression()).
   */
  bool penalty_normalized;

#ifdef __cplusplus
  qn_params()
    : loss(QN_LOSS_UNKNOWN),
      penalty_l1(0),
      penalty_l2(0),
      grad_tol(1e-4),
      change_tol(1e-5),
      max_iter(1000),
      linesearch_max_iter(50),
      lbfgs_memory(5),
      verbose(0),
      fit_intercept(true),
      penalty_normalized(true)
  {
  }
#endif
};

#ifndef __cplusplus
typedef struct qn_params qn_params;
#endif

#ifdef __cplusplus
}
}
#endif
