/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <common/cumlHandle.hpp>

namespace ML {
namespace GLM {
/**
 * @defgroup Functions fit an ordinary least squares model
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param labels        device pointer to label vector of length n_rows
 * @param coef          device pointer to hold the solution for weights of size n_cols
 * @param intercept     device pointer to hold the solution for bias term of size 1
 * @param fit_intercept if true, fit intercept
 * @param normalize     if true, normalize data to zero mean, unit variance
 * @param algo          specifies which solver to use (0: SVD, 1: Eigendecomposition, 2: QR-decomposition)
 * @{
 */
void olsFit(const cumlHandle &handle, float *input, int n_rows, int n_cols,
            float *labels, float *coef, float *intercept, bool fit_intercept,
            bool normalize, int algo = 0);
void olsFit(const cumlHandle &handle, double *input, int n_rows, int n_cols,
            double *labels, double *coef, double *intercept, bool fit_intercept,
            bool normalize, int algo = 0);
/** @} */

/**
 * @defgroup Functions fit a ridge regression model (l2 regularized least squares)
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param labels        device pointer to label vector of length n_rows
 * @param alpha         device pointer to parameters of the l2 regularizer
 * @param n_alpha       number of regularization parameters
 * @param coef          device pointer to hold the solution for weights of size n_cols
 * @param intercept     device pointer to hold the solution for bias term of size 1
 * @param fit_intercept if true, fit intercept
 * @param normalize     if true, normalize data to zero mean, unit variance
 * @param algo          specifies which solver to use (0: SVD, 1: Eigendecomposition)
 * @{
 */
void ridgeFit(const cumlHandle &handle, float *input, int n_rows, int n_cols,
              float *labels, float *alpha, int n_alpha, float *coef,
              float *intercept, bool fit_intercept, bool normalize,
              int algo = 0);

void ridgeFit(const cumlHandle &handle, double *input, int n_rows, int n_cols,
              double *labels, double *alpha, int n_alpha, double *coef,
              double *intercept, bool fit_intercept, bool normalize,
              int algo = 0);
/** @} */

/**
 * @defgroup Functions to make predictions with a fitted ordinary least squares and ridge regression model
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param coef          weights of the model
 * @param intercept     bias term of the model
 * @param preds         device pointer to store predictions of size n_rows
 * @{
 */
void olsPredict(const cumlHandle &handle, const float *input, int n_rows,
                int n_cols, const float *coef, float intercept, float *preds);
void olsPredict(const cumlHandle &handle, const double *input, int n_rows,
                int n_cols, const double *coef, double intercept,
                double *preds);

void ridgePredict(const cumlHandle &handle, const float *input, int n_rows,
                  int n_cols, const float *coef, float intercept, float *preds);

void ridgePredict(const cumlHandle &handle, const double *input, int n_rows,
                  int n_cols, const double *coef, double intercept,
                  double *preds);
/** @} */

/**
 * @defgroup functions to fit a GLM using quasi newton methods.
 * @param cuml_handle           reference to cumlHandle object
 * @param X                     device pointer to feature matrix of dimension
 * NxD (row- or column major: see X_col_major param)
 * @param y                     device pointer to label vector of length N (for
 * binary logistic: [0,1], for multinomial:  [0,...,C-1])
 * @param N                     number of examples
 * @param D                     number of features
 * @param C                     number of outputs (C > 1, for multinomial,
 * indicating number of classes. For logistic and normal, C must be 1.)
 * @param fit_intercept         true if model should include a bias. If true,
 * the initial point/result w0 should point to a memory location of size (D+1) *
 * C
 * @param l1                    l1 regularization strength (if non-zero, will
 * run OWL-QN, else L-BFGS). Note, that as in scikit, the bias will not be
 * regularized.
 * @param l2                    l2 regularization strength. Note, that as in
 * scikit, the bias will not be regularized.
 * @param max_iter              limit on iteration number
 * @param grad_tol              tolerance for gradient norm convergence check
 * @param linesearch_max_iter   max number of linesearch iterations per outer
 * iteration
 * @param lbfgs_memory          rank of the lbfgs inverse-Hessian approximation.
 * Method will request memory of size O(lbfgs_memory * D).
 * @param verbosity             verbosity level
 * @param w0                    device pointer of size (D + (fit_intercept ? 1 :
 * 0)) * C with initial point, overwritten by final result.
 * @param f                     host pointer holding the final objective value
 * @param num_iters             host pointer holding the actual number of
 * iterations taken
 * @param X_col_major           true if X is stored column-major, i.e. feature
 * columns are contiguous
 * @param loss_type             id of likelihood model (0: logistic/sigmoid, 1:
 * multinomial/softmax, 2: normal/squared)
 * @{
 */
void qnFit(const cumlHandle &cuml_handle, float *X, float *y, int N, int D,
           int C, bool fit_intercept, float l1, float l2, int max_iter,
           float grad_tol, int linesearch_max_iter, int lbfgs_memory,
           int verbosity, float *w0, float *f, int *num_iters, bool X_col_major,
           int loss_type);

void qnFit(const cumlHandle &cuml_handle, double *X, double *y, int N, int D,
           int C, bool fit_intercept, double l1, double l2, int max_iter,
           double grad_tol, int linesearch_max_iter, int lbfgs_memory,
           int verbosity, double *w0, double *f, int *num_iters,
           bool X_col_major, int loss_type);
/** @} */

/**
 * @defgroup functions to fit a GLM using quasi newton methods.
 * @param cuml_handle           reference to cumlHandle object
 * @param X                     device pointer to feature matrix of dimension NxD (row- or column major: see X_col_major param)
 * @param N                     number of examples
 * @param D                     number of features
 * @param C                     number of outputs (C > 1, for multinomial, indicating number of classes. For logistic and normal, C must be 1.)
 * @param fit_intercept         true if model includes a bias.
 * @param params                device pointer to model parameters. Length D if fit_intercept == false else D+1
 * @param X_col_major           true if X is stored column-major, i.e. feature columns are contiguous
 * @param loss_type             id of likelihood model (0: logistic/sigmoid, 1: multinomial/softmax, 2: normal/squared)
 * @param preds                 device pointer to predictions of length N (for binary logistic: [0,1], for multinomial:  [0,...,C-1])
 */
void qnPredict(const cumlHandle &cuml_handle, float *X, int N, int D, int C,
               bool fit_intercept, float *params, bool X_col_major,
               int loss_type, float *preds);

void qnPredict(const cumlHandle &cuml_handle, double *X, int N, int D, int C,
               bool fit_intercept, double *params, bool X_col_major,
               int loss_type, double *preds);
/** @} */

}  // namespace GLM
}  // namespace ML
