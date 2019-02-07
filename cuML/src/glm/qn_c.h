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
void cuml_glm_logreg_fit_sqn(float *X, float *y, int N, int D, bool has_bias,
                             float l1, float l2, int max_iter, float grad_tol,
                             float value_rel_tol, int linesearch_max_iter,
                             int lbfgs_memory, int verbosity,
                             float *w0, // initial value and result
                             float *f,  // function value
                             int *num_iters);

void cuml_glm_logreg_fit_dqn(double *X, double *y, int N, int D, bool has_bias,
                             double l1, double l2, int max_iter,
                             double grad_tol, double value_rel_tol,
                             int linesearch_max_iter, int lbfgs_memory,
                             int verbosity,
                             double *w0, // initial value and result
                             double *f, int *num_iters);

void cuml_glm_linreg_fit_sqn(float *X, float *y, int N, int D, bool has_bias,
                             float l1, float l2, int max_iter, float grad_tol,
                             float value_rel_tol, int linesearch_max_iter,
                             int lbfgs_memory, int verbosity,
                             float *w0, // initial value and result
                             float *f, int *num_iters);

void cuml_glm_linreg_fit_dqn(double *X, double *y, int N, int D, bool has_bias,
                             double l1, double l2, int max_iter,
                             double grad_tol, double value_rel_tol,
                             int linesearch_max_iter, int lbfgs_memory,
                             int verbosity,
                             double *w0, // initial value and result
                             double *f, int *num_iters);
}
