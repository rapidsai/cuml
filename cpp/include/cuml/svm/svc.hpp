/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "svm_model.h"
#include "svm_parameter.h"

#include <cuml/common/logger.hpp>
#include <cuml/matrix/kernel_params.hpp>

#include <raft/core/handle.hpp>

namespace ML {
namespace SVM {

// Forward declarations of the stateless API
/**
 * @brief Fit a support vector classifier to the training data.
 *
 * Each row of the input data stores a feature vector.
 * We use the SMO method to fit the SVM.
 *
 * The output device buffers in model shall be unallocated on entry.
 *
 * @tparam math_t floating point type
 * @param [in] handle the cuML handle
 * @param [in] input device pointer for the input data in column major format.
 *   Size n_rows x n_cols.
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of columns
 * @param [in] labels device pointer for the labels. Size [n_rows].
 * @param [in] param parameters for training
 * @param [in] kernel_params parameters for the kernel function
 * @param [out] model parameters of the trained model
 * @param [in] sample_weight optional sample weights, size [n_rows]
 */
template <typename math_t>
void svcFit(const raft::handle_t& handle,
            math_t* input,
            int n_rows,
            int n_cols,
            math_t* labels,
            const SvmParameter& param,
            ML::matrix::KernelParams& kernel_params,
            SvmModel<math_t>& model,
            const math_t* sample_weight);

/**
 * @brief Fit a support vector classifier to the training data.
 *
 * Each row of the input data stores a feature vector.
 * We use the SMO method to fit the SVM.
 *
 * The output device buffers in model shall be unallocated on entry.
 *
 * @tparam math_t floating point type
 * @param [in] handle the cuML handle
 * @param [in] indptr device pointer for CSR row positions. Size [n_rows + 1].
 * @param [in] indices device pointer for CSR column indices. Size [nnz].
 * @param [in] data device pointer for the CSR data. Size [nnz].
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of columns
 * @param [in] nnz number of stored entries.
 * @param [in] labels device pointer for the labels. Size [n_rows].
 * @param [in] param parameters for training
 * @param [in] kernel_params parameters for the kernel function
 * @param [out] model parameters of the trained model
 * @param [in] sample_weight optional sample weights, size [n_rows]
 */
template <typename math_t>
void svcFitSparse(const raft::handle_t& handle,
                  int* indptr,
                  int* indices,
                  math_t* data,
                  int n_rows,
                  int n_cols,
                  int nnz,
                  math_t* labels,
                  const SvmParameter& param,
                  ML::matrix::KernelParams& kernel_params,
                  SvmModel<math_t>& model,
                  const math_t* sample_weight);

/**
 * @brief Predict classes or decision function value for samples in input.
 *
 * We evaluate the decision function f(x_i). Depending on the parameter
 * predict_class, we either return f(x_i) or the label corresponding to
 * sign(f(x_i)).
 *
 * The predictions are calculated according to the following formulas:
 * \f[
 *    f(x_i) = \sum_{j=1}^n_support K(x_i, x_j) * dual_coefs[j] + b)
 * \f]
 *
 * pred(x_i) = label[sign(f(x_i))], if predict_class==true, or
 * pred(x_i) = f(x_i),       if predict_class==false.
 *
 * @tparam math_t floating point type
 * @param handle the cuML handle
 * @param [in] input device pointer for the input data in column major format,
 *   size [n_rows x n_cols].
 * @param [in] n_rows number of rows (input vectors)
 * @param [in] n_cols number of columns (features)
 * @param [in] kernel_params parameters for the kernel function
 * @param [in] model SVM model parameters
 * @param [out] preds device pointer to store the predicted class labels.
 *    Size [n_rows]. Should be allocated on entry.
 * @param [in] buffer_size size of temporary buffer in MiB
 * @param [in] predict_class whether to predict class label (true), or just
 *     return the decision function value (false)
 */
template <typename math_t>
void svcPredict(const raft::handle_t& handle,
                math_t* input,
                int n_rows,
                int n_cols,
                ML::matrix::KernelParams& kernel_params,
                const SvmModel<math_t>& model,
                math_t* preds,
                math_t buffer_size,
                bool predict_class);

/**
 * @brief Predict classes or decision function value for samples in input.
 *
 * We evaluate the decision function f(x_i). Depending on the parameter
 * predict_class, we either return f(x_i) or the label corresponding to
 * sign(f(x_i)).
 *
 * The predictions are calculated according to the following formulas:
 * \f[
 *    f(x_i) = \sum_{j=1}^n_support K(x_i, x_j) * dual_coefs[j] + b)
 * \f]
 *
 * pred(x_i) = label[sign(f(x_i))], if predict_class==true, or
 * pred(x_i) = f(x_i),       if predict_class==falsee.
 *
 * @tparam math_t floating point type
 * @param handle the cuML handle
 * @param [in] indptr device pointer for CSR row positions. Size [n_rows + 1].
 * @param [in] indices device pointer for CSR column indices. Size [nnz].
 * @param [in] data device pointer for the CSR data. Size [nnz].
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of columns
 * @param [in] nnz number of stored entries.
 * @param [in] kernel_params parameters for the kernel function
 * @param [in] model SVM model parameters
 * @param [out] preds device pointer to store the predicted class labels.
 *    Size [n_rows]. Should be allocated on entry.
 * @param [in] buffer_size size of temporary buffer in MiB
 * @param [in] predict_class whether to predict class label (true), or just
 *     return the decision function value (false)
 */
template <typename math_t>
void svcPredictSparse(const raft::handle_t& handle,
                      int* indptr,
                      int* indices,
                      math_t* data,
                      int n_rows,
                      int n_cols,
                      int nnz,
                      ML::matrix::KernelParams& kernel_params,
                      const SvmModel<math_t>& model,
                      math_t* preds,
                      math_t buffer_size,
                      bool predict_class);

/**
 * Deallocate device buffers in the SvmModel struct.
 *
 * @param [in] handle cuML handle
 * @param [inout] m SVM model parameters
 */
template <typename math_t>
void svmFreeBuffers(const raft::handle_t& handle, SvmModel<math_t>& m);

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

  ML::matrix::KernelParams kernel_params;
  SvmParameter param;
  SvmModel<math_t> model;
  /**
   * @brief Constructs a support vector classifier
   * @param handle cuML handle
   * @param C penalty term
   * @param tol tolerance to stop fitting
   * @param kernel_params parameters for kernels
   * @param cache_size size of kernel cache in device memory (MiB)
   * @param max_iter maximum number of outer iterations in SmoSolver
   * @param nochange_steps number of steps with no change wrt convergence
   * @param verbosity verbosity level for logging messages during execution
   */
  SVC(raft::handle_t& handle,
      math_t C   = 1,
      math_t tol = 1.0e-3,
      ML::matrix::KernelParams kernel_params =
        ML::matrix::KernelParams{ML::matrix::KernelType::LINEAR, 3, 1, 0},
      math_t cache_size                   = 200,
      int max_iter                        = -1,
      int nochange_steps                  = 1000,
      rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

  ~SVC();

  /**
   * @brief Fit a support vector classifier to the training data.
   *
   * Each row of the input data stores a feature vector.
   * We use the SMO method to fit the SVM.
   *
   * @param input device pointer for the input data in column major format. Size n_rows x n_cols.
   * @param n_rows number of rows
   * @param n_cols number of columns
   * @param labels device pointer for the labels. Size n_rows.
   * @param [in] sample_weight optional sample weights, size [n_rows]
   */
  void fit(
    math_t* input, int n_rows, int n_cols, math_t* labels, const math_t* sample_weight = nullptr);

  /**
   * @brief Predict classes for samples in input.
   * @param [in]  input device pointer for the input data in column major format,
   *   size [n_rows x n_cols].
   * @param [in] n_rows number of vectors
   * @param [in] n_cols number of features
   * @param [out] preds device pointer to store the predicted class labels.
   *    Size [n_rows]. Should be allocated on entry.
   */
  void predict(math_t* input, int n_rows, int n_cols, math_t* preds);

  /**
   * @brief Calculate decision function value for samples in input.
   * @param [in] input device pointer for the input data in column major format,
   *   size [n_rows x n_cols].
   * @param [in] n_rows number of vectors
   * @param [in] n_cols number of features
   * @param [out] preds device pointer to store the decision function value
   *    Size [n_rows]. Should be allocated on entry.
   */
  void decisionFunction(math_t* input, int n_rows, int n_cols, math_t* preds);

 private:
  const raft::handle_t& handle;
};

};  // end namespace SVM
};  // end namespace ML
