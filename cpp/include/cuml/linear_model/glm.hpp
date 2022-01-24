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

#include <raft/handle.hpp>

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
 * @{
 */
void olsFit(const raft::handle_t& handle,
            float* input,
            int n_rows,
            int n_cols,
            float* labels,
            float* coef,
            float* intercept,
            bool fit_intercept,
            bool normalize,
            int algo = 0);
void olsFit(const raft::handle_t& handle,
            double* input,
            int n_rows,
            int n_cols,
            double* labels,
            double* coef,
            double* intercept,
            bool fit_intercept,
            bool normalize,
            int algo = 0);
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
 * @{
 */
void ridgeFit(const raft::handle_t& handle,
              float* input,
              int n_rows,
              int n_cols,
              float* labels,
              float* alpha,
              int n_alpha,
              float* coef,
              float* intercept,
              bool fit_intercept,
              bool normalize,
              int algo = 0);
void ridgeFit(const raft::handle_t& handle,
              double* input,
              int n_rows,
              int n_cols,
              double* labels,
              double* alpha,
              int n_alpha,
              double* coef,
              double* intercept,
              bool fit_intercept,
              bool normalize,
              int algo = 0);
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
                 int n_rows,
                 int n_cols,
                 const float* coef,
                 float intercept,
                 float* preds);
void gemmPredict(const raft::handle_t& handle,
                 const double* input,
                 int n_rows,
                 int n_cols,
                 const double* coef,
                 double intercept,
                 double* preds);
/** @} */

/**
 * @brief Fit a GLM using quasi newton methods.
 *
 * @param cuml_handle           reference to raft::handle_t object
 * @param pams                  model parameters
 * @param X                     device pointer to feature matrix of dimension
 * @param X_col_major           true if X is stored column-major, i.e. feature
 * columns are contiguous
 * NxD (row- or column major: see X_col_major param)
 * @param y                     device pointer to label vector of length N (for
 * binary logistic: [0,1], for multinomial:  [0,...,C-1])
 * @param N                     number of examples
 * @param D                     number of features
 * @param C                     number of outputs (C > 1, for multinomial,
 * indicating number of classes. For logistic and normal, C must be 1.)
 * @param w0                    device pointer of size (D + (fit_intercept ? 1 :
 * 0)) * C with initial point, overwritten by final result.
 * @param f                     host pointer holding the final objective value
 * @param num_iters             host pointer holding the actual number of iterations taken
 * @param sample_weight
 */
template <typename T>
void qnFit(const raft::handle_t& cuml_handle,
           const qn_params& pams,
           T* X,
           bool X_col_major,
           T* y,
           int N,
           int D,
           int C,
           T* w0,
           T* f,
           int* num_iters,
           T* sample_weight = nullptr);

/**
 * @brief Fit a GLM using quasi newton methods.
 *
 * @param cuml_handle           reference to raft::handle_t object
 * @param pams                  model parameters
 * @param X_values              feature matrix values (CSR format), matrix dimension: NxD.
 * @param X_cols                feature matrix columns (CSR format)
 * @param X_row_ids             feature matrix compresses row ids (CSR format)
 * @param X_nnz                 number of non-zero entries in the feature
 * matrix (CSR format)
 * @param y                     device pointer to label vector of length N (for
 * binary logistic: [0,1], for multinomial:  [0,...,C-1])
 * @param N                     number of examples
 * @param D                     number of features
 * @param C                     number of outputs (C > 1, for multinomial,
 * indicating number of classes. For logistic and normal, C must be 1.)
 * @param w0                    device pointer of size (D + (fit_intercept ? 1 :
 * 0)) * C with initial point, overwritten by final result.
 * @param f                     host pointer holding the final objective value
 * @param num_iters             host pointer holding the actual number of iterations taken
 * @param sample_weight
 */
template <typename T>
void qnFitSparse(const raft::handle_t& cuml_handle,
                 const qn_params& pams,
                 T* X_values,
                 int* X_cols,
                 int* X_row_ids,
                 int X_nnz,
                 T* y,
                 int N,
                 int D,
                 int C,
                 T* w0,
                 T* f,
                 int* num_iters,
                 T* sample_weight = nullptr);

/**
 * @brief Obtain the confidence scores of samples
 *
 * @param cuml_handle           reference to raft::handle_t object
 * @param pams                  model parameters
 * @param X                     device pointer to feature matrix of dimension NxD (row- or
 * column-major: see X_col_major param)
 * @param X_col_major           true if X is stored column-major, i.e. feature columns are
 * contiguous
 * @param N                     number of examples
 * @param D                     number of features
 * @param C                     number of outputs (C > 1, for multinomial, indicating number of
 * classes. For logistic, C = 2 and normal, C = 1.)
 * @param params                device pointer to model parameters. Length D if fit_intercept ==
 * false else D+1
 * @param scores                device pointer to confidence scores of length N (for binary
 * logistic: [0,1], for multinomial:  [0,...,C-1])
 */
template <typename T>
void qnDecisionFunction(const raft::handle_t& cuml_handle,
                        const qn_params& pams,
                        T* X,
                        bool X_col_major,
                        int N,
                        int D,
                        int C,
                        T* params,
                        T* scores);

/**
 * @brief Obtain the confidence scores of samples
 *
 * @param cuml_handle           reference to raft::handle_t object
 * @param pams                  model parameters
 * @param X_values              feature matrix values (CSR format), matrix dimension: NxD.
 * @param X_cols                feature matrix columns (CSR format)
 * @param X_row_ids             feature matrix compresses row ids (CSR format)
 * @param X_nnz                 number of non-zero entries in the feature matrix (CSR format)
 * @param N                     number of examples
 * @param D                     number of features
 * @param C                     number of outputs (C > 1, for multinomial, indicating number of
 * classes. For logistic, C = 2 and normal, C = 1.)
 * @param params                device pointer to model parameters. Length D if fit_intercept ==
 * false else D+1
 * @param scores                device pointer to confidence scores of length N (for binary
 * logistic: [0,1], for multinomial:  [0,...,C-1])
 */
template <typename T>
void qnDecisionFunctionSparse(const raft::handle_t& cuml_handle,
                              const qn_params& pams,
                              T* X_values,
                              int* X_cols,
                              int* X_row_ids,
                              int X_nnz,
                              int N,
                              int D,
                              int C,
                              T* params,
                              T* scores);

/**
 * @brief Predict a GLM using quasi newton methods.
 *
 * @param cuml_handle           reference to raft::handle_t object
 * @param pams                  model parameters
 * @param X                     device pointer to feature matrix of dimension NxD (row- or column
 * major: see X_col_major param)
 * @param X_col_major           true if X is stored column-major, i.e. feature columns are
 * contiguous
 * @param N                     number of examples
 * @param D                     number of features
 * @param C                     number of outputs (C > 1, for multinomial, indicating number of
 * classes. For logistic and normal, C must be 1.)
 * @param params                device pointer to model parameters. Length D if fit_intercept ==
 * false else D+1
 * @param preds                 device pointer to predictions of length N (for binary logistic:
 * [0,1], for multinomial:  [0,...,C-1])
 */
template <typename T>
void qnPredict(const raft::handle_t& cuml_handle,
               const qn_params& pams,
               T* X,
               bool X_col_major,
               int N,
               int D,
               int C,
               T* params,
               T* preds);

/**
 * @brief Predict a GLM using quasi newton methods.
 *
 * @param cuml_handle           reference to raft::handle_t object
 * @param pams                  model parameters
 * @param X_values              feature matrix values (CSR format), matrix dimension: NxD.
 * @param X_cols                feature matrix columns (CSR format)
 * @param X_row_ids             feature matrix compresses row ids (CSR format)
 * @param X_nnz                 number of non-zero entries in the feature matrix (CSR format)
 * @param N                     number of examples
 * @param D                     number of features
 * @param C                     number of outputs (C > 1, for multinomial, indicating number of
 * classes. For logistic and normal, C must be 1.)
 * @param params                device pointer to model parameters. Length D if fit_intercept ==
 * false else D+1
 * @param preds                 device pointer to predictions of length N (for binary logistic:
 * [0,1], for multinomial:  [0,...,C-1])
 */
template <typename T>
void qnPredictSparse(const raft::handle_t& cuml_handle,
                     const qn_params& pams,
                     T* X_values,
                     int* X_cols,
                     int* X_row_ids,
                     int X_nnz,
                     int N,
                     int D,
                     int C,
                     T* params,
                     T* preds);

}  // namespace GLM
}  // namespace ML
