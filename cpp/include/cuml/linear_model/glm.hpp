/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <cuml/linear_model/qn.h>

#include <raft/core/handle.hpp>

namespace ML {
namespace GLM {

/**
 * @defgroup olsFit fit an ordinary least squares model
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param labels        device pointer to label vector of length n_rows
 * @param coef          device pointer to hold the solution for weights of size n_cols
 * @param intercept     host pointer to hold the solution for bias term of size 1
 * @param fit_intercept if true, fit intercept
 * @param normalize     if true, normalize data to zero mean, unit variance
 * @param algo          specifies which solver to use (0: SVD, 1: Eigendecomposition, 2:
 * QR-decomposition)
 * @param sample_weight device pointer to sample weight vector of length n_rows (nullptr
   for uniform weights) This vector is modified during the computation
 * @{
 */
void olsFit(const raft::handle_t& handle,
            float* input,
            size_t n_rows,
            size_t n_cols,
            float* labels,
            float* coef,
            float* intercept,
            bool fit_intercept,
            bool normalize,
            int algo             = 0,
            float* sample_weight = nullptr);
void olsFit(const raft::handle_t& handle,
            double* input,
            size_t n_rows,
            size_t n_cols,
            double* labels,
            double* coef,
            double* intercept,
            bool fit_intercept,
            bool normalize,
            int algo              = 0,
            double* sample_weight = nullptr);
/** @} */

/**
 * @defgroup ridgeFit fit a ridge regression model (l2 regularized least squares)
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param labels        device pointer to label vector of length n_rows
 * @param alpha         host pointer to parameters of the l2 regularizer
 * @param n_alpha       number of regularization parameters
 * @param coef          device pointer to hold the solution for weights of size n_cols
 * @param intercept     host pointer to hold the solution for bias term of size 1
 * @param fit_intercept if true, fit intercept
 * @param normalize     if true, normalize data to zero mean, unit variance
 * @param algo          specifies which solver to use (0: SVD, 1: Eigendecomposition)
 * @param sample_weight device pointer to sample weight vector of length n_rows (nullptr
   for uniform weights) This vector is modified during the computation
 * @{
 */
void ridgeFit(const raft::handle_t& handle,
              float* input,
              size_t n_rows,
              size_t n_cols,
              float* labels,
              float* alpha,
              int n_alpha,
              float* coef,
              float* intercept,
              bool fit_intercept,
              bool normalize,
              int algo             = 0,
              float* sample_weight = nullptr);
void ridgeFit(const raft::handle_t& handle,
              double* input,
              size_t n_rows,
              size_t n_cols,
              double* labels,
              double* alpha,
              int n_alpha,
              double* coef,
              double* intercept,
              bool fit_intercept,
              bool normalize,
              int algo              = 0,
              double* sample_weight = nullptr);
/** @} */

/**
 * @defgroup glmPredict to make predictions with a fitted ordinary least squares and ridge
 * regression model
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param coef          weights of the model
 * @param intercept     bias term of the model
 * @param preds         device pointer to store predictions of size n_rows
 * @{
 */
void gemmPredict(const raft::handle_t& handle,
                 const float* input,
                 size_t n_rows,
                 size_t n_cols,
                 const float* coef,
                 float intercept,
                 float* preds);
void gemmPredict(const raft::handle_t& handle,
                 const double* input,
                 size_t n_rows,
                 size_t n_cols,
                 const double* coef,
                 double intercept,
                 double* preds);
/** @} */

/**
 * @brief Fit a GLM using quasi newton methods.
 *
 * @param cuml_handle   reference to raft::handle_t object
 * @param params        model parameters
 * @param X             device pointer to a contiguous feature matrix of dimension [N, D]
 * @param X_col_major   true if X is stored column-major
 * @param y             device pointer to label vector of length N
 * @param N             number of examples
 * @param D             number of features
 * @param C             number of outputs (number of classes or `1` for regression)
 * @param w0            device pointer of size (D + (fit_intercept ? 1 : 0)) * C with initial point,
 *                      overwritten by final result.
 * @param f             host pointer holding the final objective value
 * @param num_iters     host pointer holding the actual number of iterations taken
 * @param sample_weight device pointer to sample weight vector of length n_rows (nullptr
   for uniform weights)
 * @param svr_eps       epsilon parameter for svr
 */
template <typename T, typename I = int>
void qnFit(const raft::handle_t& cuml_handle,
           const qn_params& params,
           T* X,
           bool X_col_major,
           T* y,
           I N,
           I D,
           I C,
           T* w0,
           T* f,
           int* num_iters,
           T* sample_weight = nullptr,
           T svr_eps        = 0);

/**
 * @brief Fit a GLM using quasi newton methods.
 *
 * @param cuml_handle   reference to raft::handle_t object
 * @param params        model parameters
 * @param X_values      feature matrix values (CSR format), length = X_nnz
 * @param X_cols        feature matrix columns (CSR format), length = X_nnz, range = [0, ... D-1]
 * @param X_row_ids     feature matrix compressed row ids (CSR format),
 *                      length = N + 1, range = [0, ... X_nnz]
 * @param X_nnz         number of non-zero entries in the feature matrix (CSR format)
 * @param y             device pointer to label vector of length N
 * @param N             number of examples
 * @param D             number of features
 * @param C             number of outputs (number of classes or `1` for regression)
 * @param w0            device pointer of size (D + (fit_intercept ? 1 : 0)) * C with initial point,
 *                      overwritten by final result.
 * @param f             host pointer holding the final objective value
 * @param num_iters     host pointer holding the actual number of iterations taken
 * @param sample_weight device pointer to sample weight vector of length n_rows (nullptr
   for uniform weights)
 * @param svr_eps       epsilon parameter for svr
 */
template <typename T, typename I = int>
void qnFitSparse(const raft::handle_t& cuml_handle,
                 const qn_params& params,
                 T* X_values,
                 I* X_cols,
                 I* X_row_ids,
                 I X_nnz,
                 T* y,
                 I N,
                 I D,
                 I C,
                 T* w0,
                 T* f,
                 int* num_iters,
                 T* sample_weight = nullptr,
                 T svr_eps        = 0);

/**
 * @brief Obtain the confidence scores of samples
 *
 * @param cuml_handle   reference to raft::handle_t object
 * @param params        model parameters
 * @param X             device pointer to a contiguous feature matrix of dimension [N, D]
 * @param X_col_major   true if X is stored column-major
 * @param N             number of examples
 * @param D             number of features
 * @param C             number of outputs (number of classes or `1` for regression)
 * @param coefs         device pointer to model coefficients. Length D if fit_intercept == false
 *                      else D+1
 * @param scores        device pointer to confidence scores of length N (for binary logistic: [0,1],
 *                      for multinomial:  [0,...,C-1])
 */
template <typename T, typename I = int>
void qnDecisionFunction(const raft::handle_t& cuml_handle,
                        const qn_params& params,
                        T* X,
                        bool X_col_major,
                        I N,
                        I D,
                        I C,
                        T* coefs,
                        T* scores);

/**
 * @brief Obtain the confidence scores of samples
 *
 * @param cuml_handle   reference to raft::handle_t object
 * @param params        model parameters
 * @param X_values      feature matrix values (CSR format), length = X_nnz
 * @param X_cols        feature matrix columns (CSR format), length = X_nnz, range = [0, ... D-1]
 * @param X_row_ids     feature matrix compressed row ids (CSR format),
 *                      length = N + 1, range = [0, ... X_nnz]
 * @param X_nnz         number of non-zero entries in the feature matrix (CSR format)
 * @param N             number of examples
 * @param D             number of features
 * @param C             number of outputs (number of classes or `1` for regression)
 * @param coefs         device pointer to model coefficients. Length D if fit_intercept == false
 *                      else D+1
 * @param scores        device pointer to confidence scores of length N (for binary logistic: [0,1],
 *                      for multinomial:  [0,...,C-1])
 */
template <typename T, typename I = int>
void qnDecisionFunctionSparse(const raft::handle_t& cuml_handle,
                              const qn_params& params,
                              T* X_values,
                              I* X_cols,
                              I* X_row_ids,
                              I X_nnz,
                              I N,
                              I D,
                              I C,
                              T* coefs,
                              T* scores);

/**
 * @brief Predict a GLM using quasi newton methods.
 *
 * @param cuml_handle   reference to raft::handle_t object
 * @param params        model parameters
 * @param X             device pointer to a contiguous feature matrix of dimension [N, D]
 * @param X_col_major   true if X is stored column-major
 * @param N             number of examples
 * @param D             number of features
 * @param C             number of outputs (number of classes or `1` for regression)
 * @param coefs         device pointer to model coefficients. Length D if fit_intercept == false
 *                      else D+1
 * @param preds         device pointer to predictions of length N (for binary logistic: [0,1],
 *                      for multinomial:  [0,...,C-1])
 */
template <typename T, typename I = int>
void qnPredict(const raft::handle_t& cuml_handle,
               const qn_params& params,
               T* X,
               bool X_col_major,
               I N,
               I D,
               I C,
               T* coefs,
               T* preds);

/**
 * @brief Predict a GLM using quasi newton methods.
 *
 * @param cuml_handle   reference to raft::handle_t object
 * @param params        model parameters
 * @param X_values      feature matrix values (CSR format), length = X_nnz
 * @param X_cols        feature matrix columns (CSR format), length = X_nnz, range = [0, ... D-1]
 * @param X_row_ids     feature matrix compressed row ids (CSR format),
 *                      length = N + 1, range = [0, ... X_nnz]
 * @param X_nnz         number of non-zero entries in the feature matrix (CSR format)
 * @param N             number of examples
 * @param D             number of features
 * @param C             number of outputs (number of classes or `1` for regression)
 * @param coefs         device pointer to model coefficients. Length D if fit_intercept == false
 *                      else D+1
 * @param preds         device pointer to predictions of length N (for binary logistic: [0,1],
 *                      for multinomial:  [0,...,C-1])
 */
template <typename T, typename I = int>
void qnPredictSparse(const raft::handle_t& cuml_handle,
                     const qn_params& params,
                     T* X_values,
                     I* X_cols,
                     I* X_row_ids,
                     I X_nnz,
                     I N,
                     I D,
                     I C,
                     T* coefs,
                     T* preds);

}  // namespace GLM
}  // namespace ML
