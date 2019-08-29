/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cublas_v2.h>
#include "common/cumlHandle.hpp"
#include "gram/kernelparams.h"
#include "svm_model.h"
#include "svm_parameter.h"

namespace ML {
namespace SVM {

// Forward declarations of the stateless API
template <typename math_t>
void svcFit(const cumlHandle &handle, math_t *input, int n_rows, int n_cols,
            math_t *labels, const svmParameter &param,
            MLCommon::GramMatrix::KernelParams &kernel_params,
            svmModel<math_t> &model);

template <typename math_t>
void svcPredict(const cumlHandle &handle, math_t *input, int n_rows, int n_cols,
                MLCommon::GramMatrix::KernelParams &kernel_params,
                const svmModel<math_t> &model, math_t *preds);

template <typename math_t>
void svmFreeBuffers(const cumlHandle &handle, svmModel<math_t> &m);

/**
 * @brief C-Support Vector Classification
 *
 * This is a Scikit-Learn like wrapper around the stateless C++ functions.
 * See Issue #456 for general discussion about stateful Sklearn like wrappers.
 *
 * The classifier will be fitted using the SMO algorithm in dual space.
 *
 * The decision function takes the following form
 * \f[
 *    sign\left( \sum_{i=1}^{N_{support}} y_i \alpha_i K(x_i,x) + b \right),
 * \f]
 * where \f$x_i\f$ are the support vectors, and \f$ y_i \alpha_i \f$ are the dual
 * coordinates.
 *
 * The penalty parameter C limits the values of the dual coefficients
 * \f[ 0 <= \alpha <= C \f]
 *
 */
template <typename math_t>
class SVC {
 public:
  // Public members for easier access during testing from Python.

  MLCommon::GramMatrix::KernelParams kernel_params;
  svmParameter param;
  svmModel<math_t> model;
  /**
   * @brief Constructs a support vector classifier
   * @param handle cuML handle
   * @param C penalty term
   * @param tol tolerance to stop fitting
   * @param kernel_params parameters for kernels
   * @param cache_size size of kernel cache in device memory (MiB)
   * @param max_iter maximum number of outer iterations in SmoSolver
   * @param verbose enable verbose output
   */
  SVC(cumlHandle &handle, math_t C = 1, math_t tol = 1.0e-3,
      MLCommon::GramMatrix::KernelParams kernel_params =
        MLCommon::GramMatrix::KernelParams{MLCommon::GramMatrix::LINEAR, 3, 1,
                                           0},
      math_t cache_size = 200, int max_iter = -1, bool verbose = false);

  ~SVC();

  /**
   * @brief Fit a support vector classifier to the training data.
   *
   * Each row of the input data stores a feature vector.
   * We use the SMO method to fit the SVM.
   *
   * @param input device pointer for the input data in column major format. Size n_rows x n_cols.
   * @param n_rows number of rows
   * @param n_cols number of colums
   * @param labels device pointer for the labels. Size n_rows.
   */
  void fit(math_t *input, int n_rows, int n_cols, math_t *labels);

  /**
   * @brief Predict classes for samples in input.
   * @param [in]  input device pointer for the input data in column major format,
   *   size [n_rows x n_cols].
   * @param [in] n_rows, number of vectors
   * @param [in] n_cols number of featurs
   * @param [out] preds device pointer to store the predicted class labels.
   *    Size [n_rows]. Should be allocated on entry.
   */
  void predict(math_t *input, int n_rows, int n_cols, math_t *preds);

 private:
  const cumlHandle &handle;
};

};  // end namespace SVM
};  // end namespace ML
