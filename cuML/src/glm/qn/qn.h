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
#include <glm/qn/glm_base.h>
#include <glm/qn/glm_linear.h>
#include <glm/qn/glm_logistic.h>
#include <glm/qn/glm_regularizer.h>
#include <glm/qn/glm_softmax.h>
#include <glm/qn/qn_solvers.h>


namespace ML {
namespace GLM {
template <typename T, typename LossFunction>
int qn_fit(LossFunction &loss, T *Xptr, T *yptr, T *zptr, int N,
           bool fit_intercept, T l1, T l2, int max_iter, T grad_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity,
           T *w0, // initial value and result
           T *fx, int *num_iters, STORAGE_ORDER ordX, cudaStream_t stream) {

  LBFGSParam<T> opt_param;
  opt_param.epsilon = grad_tol;
  opt_param.max_iterations = max_iter;
  opt_param.m = lbfgs_memory;
  opt_param.max_linesearch = linesearch_max_iter;
  SimpleVec<T> w(w0, loss.n_param);

  if (l2 == 0) {
    GLMWithData<T, LossFunction> lossWith(&loss, Xptr, yptr, zptr, N, ordX);

    return qn_minimize(w, fx, num_iters, lossWith, l1, opt_param, verbosity,
                       stream);

  } else {

    Tikhonov<T> reg(l2);
    RegularizedGLM<T, LossFunction, decltype(reg)> obj(&loss, &reg);
    GLMWithData<T, decltype(obj)> lossWith(&obj, Xptr, yptr, zptr, N, ordX);

    return qn_minimize(w, fx, num_iters, lossWith, l1, opt_param, verbosity,
                       stream);
  }
}

template <typename T>
void qnFit(T *X, T *y, int N, int D, int C, bool fit_intercept, T l1, T l2,
           int max_iter, T grad_tol, int linesearch_max_iter, int lbfgs_memory,
           int verbosity, T *w0, T *f, int *num_iters, bool X_col_major,
           int loss_type, cublasHandle_t cublas, cudaStream_t stream) {

  // TODO this will come from the cuml handle

  STORAGE_ORDER ord = X_col_major ? COL_MAJOR : ROW_MAJOR;

  SimpleMatOwning<T> z(C, N); // TODO this allocation somewhere else?

  switch (loss_type) {
  case 0: {
    ASSERT(C == 1, "qn.h: logistic loss invalid C");
    LogisticLoss<T> loss(D, fit_intercept, cublas);
    qn_fit<T, decltype(loss)>(loss, X, y, z.data, N, fit_intercept, l1, l2,
                               max_iter, grad_tol, linesearch_max_iter,
                               lbfgs_memory, verbosity, w0, f, num_iters, ord,
                               stream);
  } break;
  case 1: {

    ASSERT(C == 1, "qn.h: squared loss invalid C");
    SquaredLoss<T> loss(D, fit_intercept, cublas);
    qn_fit<T, decltype(loss)>(loss, X, y, z.data, N, fit_intercept, l1, l2,
                               max_iter, grad_tol, linesearch_max_iter,
                               lbfgs_memory, verbosity, w0, f, num_iters, ord,
                               stream);
  } break;
  case 2: {

    ASSERT(C > 1, "qn.h: softmax invalid C");
    Softmax<T> loss(D, C, fit_intercept, cublas);
    qn_fit<T, decltype(loss)>(loss, X, y, z.data, N, fit_intercept, l1, l2,
                               max_iter, grad_tol, linesearch_max_iter,
                               lbfgs_memory, verbosity, w0, f, num_iters, ord,
                               stream);
  } break;
  default: { ASSERT(false, "qn.h: unknown loss function."); }
  }
}

}; // namespace GLM
}; // namespace ML
