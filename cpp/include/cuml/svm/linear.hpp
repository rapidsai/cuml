/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/export.hpp>
#include <cuml/common/logger.hpp>

#include <raft/core/handle.hpp>

namespace CUML_EXPORT ML {
namespace SVM {
namespace linear {

struct Params {
  /** The regularization term. */
  enum Penalty {
    /** Abs. value of the weights: `sum |w|` */
    L1,
    /** Squared value of the weights: `sum w^2` */
    L2
  };
  /** The loss function. */
  enum Loss {
    /** `max(1 - y_i x_i w, 0)` */
    HINGE,
    /** `max(1 - y_i x_i w, 0)^2` */
    SQUARED_HINGE,
    /** `max(|y_i - x_i w| - epsilon, 0)` */
    EPSILON_INSENSITIVE,
    /** `max(|y_i - x_i w| - epsilon, 0)^2` */
    SQUARED_EPSILON_INSENSITIVE
  };

  /** The regularization term. */
  Penalty penalty = L2;
  /** The loss function. */
  Loss loss = HINGE;
  /** Whether to fit the bias term. */
  bool fit_intercept = true;
  /** When true, the bias term is treated the same way as other data features.
   *  Enabling this feature forces an extra copying the input data X.
   */
  bool penalized_intercept = false;
  /** Maximum number of iterations for the underlying QN solver. */
  int max_iter = 1000;
  /**
   * Maximum number of linesearch (inner loop) iterations for the underlying QN solver.
   */
  int linesearch_max_iter = 100;
  /**
   * Number of vectors approximating the hessian for the underlying QN solver (l-bfgs).
   */
  int lbfgs_memory = 5;
  /** Triggers extra output when greater than zero. */
  rapids_logger::level_enum verbose = rapids_logger::level_enum::off;
  /**
   * The constant scaling factor of the main term in the loss function.
   * (You can also think of that as the inverse factor of the penalty term).
   */
  double C = 1.0;
  /** The threshold on the gradient for the underlying QN solver. */
  double grad_tol = 0.0001;
  /** The threshold on the function change for the underlying QN solver. */
  double change_tol = 0.00001;
  /** The epsilon-sensitivity parameter (applicable to the SVM-regression (SVR) loss functions). */
  double epsilon = 0.0;
};

/**
 * @brief Fit a linear SVM model.
 *
 * @param [in] handle: the cuML handle.
 * @param [in] params: the model parameters.
 * @param [in] nRows: the number of input samples.
 * @param [in] nCols: the number of feature dimensions.
 * @param [in] nClasses: the number of input classes, or 0 for a regression problem.
 * @param [in] classes: the unique input classes, shape=(nClasses,), or nullptr
 * for a regression problem.
 * @param [in] X: the training data, shape=(nRows, nCols), F-contiguous
 * @param [in] y: the target data, shape=(nRows,)
 * @param [in] sampleWeight: non-negative weights for the training data, shape=(nRows,),
 * or nullptr if unweighted.
 * @param [out] w: the fitted weights, shape=(nCoefs, nCols) or (nCoefs + 1, nCols + 1)
 * if `fit_intercept=true`, where nCoefs = 1 for regression or if nClasses = 2, and
 * nClasses otherwise. F-contiguous.
 * @return n_iter: the maximum number of iterations run across all classes.
 */
template <typename T>
int fit(const raft::handle_t& handle,
        const Params& params,
        const std::size_t nRows,
        const std::size_t nCols,
        const int nClasses,
        const T* classes,
        const T* X,
        const T* y,
        const T* sampleWeight,
        T* w);
}  // namespace linear
}  // namespace SVM
}  // namespace CUML_EXPORT ML
