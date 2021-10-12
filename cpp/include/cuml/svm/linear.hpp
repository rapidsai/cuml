/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

/**
 * @file linear_svm.cuh
 * @brief Fit linear SVM.
 */

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

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
    /** `max(|y_i - x_i w| - svr_sensitivity, 0)` */
    EPSILON_INSENSITIVE,
    /** `max(|y_i - x_i w| - svr_sensitivity, 0)^2` */
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
  int verbose = 0;
  /**
   * The constant scaling factor of the main term in the loss function.
   * (You can also think of that as the inverse factor of the penalty term).
   */
  double C = 1.0;
  /** The threshold on the gradient for the underlying QN solver. */
  double grad_tol = 0.0001;
  /** The threshold on the function change for the underlying QN solver. */
  double change_tol = 0.0001;
  /** The epsilon-sensitivity parameter (applicable to the SVM-regression (SVR) loss functions). */
  double svr_sensitivity = 0.0;
  /** The value considered 'one' in the binary classification problem
   *  (applicable to the SVM-classification (SVC) loss functions).
   *  This value is converted into `1.0` during training, whereas all the other values
   *  in the training target data (`y`) are converted into `-1.0`.
   */
  double H1_value = 1.0;
};

template <typename T>
class LinearSVMModel {
 public:
  const LinearSVMParams params;
  const raft::handle_t& handle;
  const int nRows;
  const int nCols;
  /** Always contains nCols + 1 values, where the last value is the bias (possibly zero). */
  rmm::device_uvector<T> w;

  LinearSVMModel(const raft::handle_t& handle,
                 const LinearSVMParams params,
                 const T* X,
                 const int nRows,
                 const int nCols,
                 const T* y,
                 const T* sampleWeight);

  void predict(const T* X, const int nRows, const int nCols, T* out) const;

  T getIntercept() const
  {
    if (params.fit_intercept) return w.back_element(handle.get_stream());
    return 0;
  }

  T* getCoefsPtr() { return w.data(); }

  int getCoefsCount() const { return w.size() - 1; }
};

}  // namespace SVM
}  // namespace ML
