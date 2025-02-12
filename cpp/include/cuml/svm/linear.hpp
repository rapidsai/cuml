/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>

#include <raft/core/handle.hpp>

namespace ML {
namespace SVM {

struct LinearSVMParams {
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
  /** Whether to estimate probabilities using Platt scaling (applicable to SVC). */
  bool probability = false;
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

template <typename T>
struct LinearSVMModel {
  /**
   * C-style (row-major) matrix of coefficients of size `(coefRows, coefCols)`
   * where
   *   coefRows = nCols + (params.fit_intercept ? 1 : 0)
   *   coefCols = nClasses == 2 ? 1 : nClasses
   */
  T* w;
  /** Sorted, unique values of input array `y`. */
  T* classes = nullptr;
  /**
   * C-style (row-major) matrix of the probabolistic model calibration coefficients.
   * It's empty if `LinearSVMParams.probability == false`.
   * Otherwise, it's size is `(2, coefCols)`.
   * where
   *   coefCols = nClasses == 2 ? 1 : nClasses
   */
  T* probScale = nullptr;
  /** Number of classes (not applicable for regression). */
  std::size_t nClasses = 0;
  /** Number of rows of `w`, which is the number of data features plus maybe bias. */
  std::size_t coefRows;

  /** It's 1 for binary classification or regression; nClasses for multiclass. */
  inline std::size_t coefCols() const { return nClasses <= 2 ? 1 : nClasses; }

  /**
   * @brief Allocate and fit the LinearSVM model.
   *
   * @param [in] handle the cuML handle.
   * @param [in] params the model parameters.
   * @param [in] X the input data matrix of size (nRows, nCols) in column-major format.
   * @param [in] nRows the number of input samples.
   * @param [in] nCols the number of feature dimensions.
   * @param [in] y the target - a single vector of either real (regression) or
   *               categorical (classification) values (nRows, ).
   * @param [in] sampleWeight the non-negative weights for the training sample (nRows, ).
   * @return the trained model (don't forget to call `free` on it after use).
   */
  static LinearSVMModel<T> fit(const raft::handle_t& handle,
                               const LinearSVMParams& params,
                               const T* X,
                               const std::size_t nRows,
                               const std::size_t nCols,
                               const T* y,
                               const T* sampleWeight);

  /**
   * @brief Explicitly allocate the data for the model without training it.
   *
   * @param [in] handle the cuML handle.
   * @param [in] params the model parameters.
   * @param [in] nCols the number of feature dimensions.
   * @param [in] nClasses the number of classes in the dataset (not applicable for regression).
   * @return the trained model (don't forget to call `free` on it after use).
   */
  static LinearSVMModel<T> allocate(const raft::handle_t& handle,
                                    const LinearSVMParams& params,
                                    const std::size_t nCols,
                                    const std::size_t nClasses = 0);

  /** @brief Free the allocated memory. The model is not usable after the call of this method. */
  static void free(const raft::handle_t& handle, LinearSVMModel<T>& model);

  /**
   * @brief Predict using the trained LinearSVM model.
   *
   * @param [in] handle the cuML handle.
   * @param [in] params the model parameters.
   * @param [in] model the trained model.
   * @param [in] X the input data matrix of size (nRows, nCols) in column-major format.
   * @param [in] nRows the number of input samples.
   * @param [in] nCols the number of feature dimensions.
   * @param [out] out the predictions (nRows, ).
   */
  static void predict(const raft::handle_t& handle,
                      const LinearSVMParams& params,
                      const LinearSVMModel<T>& model,
                      const T* X,
                      const std::size_t nRows,
                      const std::size_t nCols,
                      T* out);

  /**
   * @brief Calculate decision function value for samples in input.
   * @param [in] handle the cuML handle.
   * @param [in] params the model parameters.
   * @param [in] model the trained model.
   * @param [in] X the input data matrix of size (nRows, nCols) in column-major format.
   * @param [in] nRows number of vectors
   * @param [in] nCols number of features
   * @param [out] out the decision function value of size (nRows, n_classes <= 2 ? 1 : n_classes) in
   * row-major format.
   */
  static void decisionFunction(const raft::handle_t& handle,
                               const LinearSVMParams& params,
                               const LinearSVMModel<T>& model,
                               const T* X,
                               const std::size_t nRows,
                               const std::size_t nCols,
                               T* out);

  /**
   * @brief For SVC, predict the probabilities for each outcome.
   *
   * @param [in] handle the cuML handle.
   * @param [in] params the model parameters.
   * @param [in] model the trained model.
   * @param [in] X the input data matrix of size (nRows, nCols) in column-major format.
   * @param [in] nRows the number of input samples.
   * @param [in] nCols the number of feature dimensions.
   * @param [in] log whether to output log-probabilities instead of probabilities.
   * @param [out] out the estimated probabilities (nRows, nClasses) in row-major format.
   */
  static void predictProba(const raft::handle_t& handle,
                           const LinearSVMParams& params,
                           const LinearSVMModel<T>& model,
                           const T* X,
                           const std::size_t nRows,
                           const std::size_t nCols,
                           const bool log,
                           T* out);
};

}  // namespace SVM
}  // namespace ML
