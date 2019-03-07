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

extern "C" {
// train regularized GLMs using L-BFGS-type quasi newton algorithms
// if l1 > 0, we will use OWL-QN, an extension to L-BFGS

/**
 * @defgroup functions to fit a GLM using quasi newton methods.
 * @param X: pointer to feature matrix of dimension NxD
 * @param y: pointer to label vector of length N
 * @param N: number of examples
 * @param D: number of features
 * @param C: number of classes (softmax only)
 * @param fit_intercept: should model include a bias?
 * @param l1: l1 regularization strength
 * @param l2: l2 regularization strength
 * @param max_iter: limit on iteration number
 * @param grad_tol: tolerance for convergence check
 * @param linesearch_max_iter: max number of linesearch iterations
 * @param lbfgs_memory: rank of the lbfgs inv-Hessian approximation
 * @param verbosity: verbosity level
 * @param w0: vector of length (D + fit_intercept ) * C with starting point and final result
 * @param f: host pointer holding the final objective value
 * @param num_iters: host pointer holding the actual number of iterations taken
 * @param X_col_major: true if X is stored column-major, i.e. feature column contiguous
 * @{
 */
void cuml_glm_fit_qn_logistic_s(float *X, float *y, int N, int D,
                                bool fit_intercept, float l1, float l2,
                                int max_iter, float grad_tol,
                                int linesearch_max_iter, int lbfgs_memory,
                                int verbosity,
                                float *w0, // initial value and result
                                float *f,  // function value
                                int *num_iters, bool X_col_major);

void cuml_glm_fit_qn_logistic_d(double *X, double *y, int N, int D,
                                bool fit_intercept, double l1, double l2,
                                int max_iter, double grad_tol,
                                int linesearch_max_iter, int lbfgs_memory,
                                int verbosity,
                                double *w0, // initial value and result
                                double *f, int *num_iters, bool X_col_major);

void cuml_glm_fit_qn_linear_s(float *X, float *y, int N, int D,
                              bool fit_intercept, float l1, float l2,
                              int max_iter, float grad_tol,
                              int linesearch_max_iter, int lbfgs_memory,
                              int verbosity,
                              float *w0, // initial value and result
                              float *f, int *num_iters, bool X_col_major);

void cuml_glm_fit_qn_linear_d(double *X, double *y, int N, int D,
                              bool fit_intercept, double l1, double l2,
                              int max_iter, double grad_tol,
                              int linesearch_max_iter, int lbfgs_memory,
                              int verbosity,
                              double *w0, // initial value and result
                              double *f, int *num_iters, bool X_col_major);

void cuml_glm_fit_qn_softmax_s(float *X, float *y, int N, int D, int C,
                               bool fit_intercept, float l1, float l2,
                               int max_iter, float grad_tol,
                               int linesearch_max_iter, int lbfgs_memory,
                               int verbosity,
                               float *w0, // initial value and result
                               float *f, int *num_iters, bool X_col_major);

void cuml_glm_fit_qn_softmax_d(double *X, double *y, int N, int D, int C,
                               bool fit_intercept, double l1, double l2,
                               int max_iter, double grad_tol,
                               int linesearch_max_iter, int lbfgs_memory,
                               int verbosity,
                               double *w0, // initial value and result
                               double *f, int *num_iters, bool X_col_major);
/** @} */
}
