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

#include <cstdio>
#include <glm/glm_base.h>
#include <glm/glm_linear.h>
#include <glm/glm_logistic.h>
#include <glm/glm_regularizer.h>
#include <glm/glm_softmax.h>
#include <glm/qn_c.h>
#include <optim/qn_solvers.h>

/*
 * Implementation of the quasi newton C api
 */

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

void logisticFitQN(double *X, double *y, int N, int D, bool fit_intercept,
                   double l1, double l2, int max_iter, double grad_tol,
                   int linesearch_max_iter, int lbfgs_memory, int verbosity,
                   double *w0, double *f, int *num_iters, bool X_col_major) {

  // TODO this will come from the cuml handle
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  cudaStream_t stream = 0;

  typedef double Real;
  typedef LogisticLoss<Real> LossFunction;

  LossFunction loss(D, fit_intercept, cublas);
  SimpleVecOwning<Real> z(N); // TODO this allocation somewhere else?
  STORAGE_ORDER ord = X_col_major ? COL_MAJOR : ROW_MAJOR;

  qn_fit<Real, LossFunction>(loss, X, y, z.data, N, fit_intercept, l1, l2,
                             max_iter, grad_tol, linesearch_max_iter,
                             lbfgs_memory, verbosity, w0, f, num_iters, ord,
                             stream);
}

void logisticFitQN(float *X, float *y, int N, int D, bool fit_intercept,
                   float l1, float l2, int max_iter, float grad_tol,
                   int linesearch_max_iter, int lbfgs_memory, int verbosity,
                   float *w0, float *f, int *num_iters, bool X_col_major) {

  // TODO this will come from the cuml handle
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  cudaStream_t stream = 0;

  typedef float Real;
  typedef LogisticLoss<Real> LossFunction;

  LossFunction loss(D, fit_intercept, cublas);
  SimpleVecOwning<Real> z(N); // TODO this allocation somewhere else?
  STORAGE_ORDER ord = X_col_major ? COL_MAJOR : ROW_MAJOR;

  qn_fit<Real, LossFunction>(loss, X, y, z.data, N, fit_intercept, l1, l2,
                             max_iter, grad_tol, linesearch_max_iter,
                             lbfgs_memory, verbosity, w0, f, num_iters, ord,
                             stream);
}

void linearFitQN(double *X, double *y, int N, int D, bool fit_intercept,
                 double l1, double l2, int max_iter, double grad_tol,
                 int linesearch_max_iter, int lbfgs_memory, int verbosity,
                 double *w0, double *f, int *num_iters, bool X_col_major) {

  // TODO this will come from the cuml handle
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  cudaStream_t stream = 0;

  typedef double Real;
  typedef SquaredLoss<Real> LossFunction;

  LossFunction loss(D, fit_intercept, cublas);
  SimpleVecOwning<Real> z(N); // TODO this allocation somewhere else?
  STORAGE_ORDER ord = X_col_major ? COL_MAJOR : ROW_MAJOR;

  qn_fit<Real, LossFunction>(loss, X, y, z.data, N, fit_intercept, l1, l2,
                             max_iter, grad_tol, linesearch_max_iter,
                             lbfgs_memory, verbosity, w0, f, num_iters, ord,
                             stream);
}

void linearFitQN(float *X, float *y, int N, int D, bool fit_intercept, float l1,
                 float l2, int max_iter, float grad_tol,
                 int linesearch_max_iter, int lbfgs_memory, int verbosity,
                 float *w0, float *f, int *num_iters, bool X_col_major) {

  // TODO this will come from the cuml handle
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  cudaStream_t stream = 0;

  typedef float Real;
  typedef SquaredLoss<Real> LossFunction;

  LossFunction loss(D, fit_intercept, cublas);
  SimpleVecOwning<Real> z(N); // TODO this allocation somewhere else?
  STORAGE_ORDER ord = X_col_major ? COL_MAJOR : ROW_MAJOR;

  qn_fit<Real, LossFunction>(loss, X, y, z.data, N, fit_intercept, l1, l2,
                             max_iter, grad_tol, linesearch_max_iter,
                             lbfgs_memory, verbosity, w0, f, num_iters, ord,
                             stream);
}

void softmaxFitQN(double *X, double *y, int N, int D, int C, bool fit_intercept,
                  double l1, double l2, int max_iter, double grad_tol,
                  int linesearch_max_iter, int lbfgs_memory, int verbosity,
                  double *w0, double *f, int *num_iters, bool X_col_major) {

  // TODO this will come from the cuml handle
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  cudaStream_t stream = 0;

  typedef double Real;
  typedef Softmax<Real> LossFunction;

  LossFunction loss(D, C, fit_intercept, cublas);
  SimpleMatOwning<Real> z(C, N); // TODO this allocation somewhere else?
  STORAGE_ORDER ord = X_col_major ? COL_MAJOR : ROW_MAJOR;

  qn_fit<Real, LossFunction>(loss, X, y, z.data, N, fit_intercept, l1, l2,
                             max_iter, grad_tol, linesearch_max_iter,
                             lbfgs_memory, verbosity, w0, f, num_iters, ord,
                             stream);
}

void softmaxFitQN(float *X, float *y, int N, int D, int C, bool fit_intercept,
                  float l1, float l2, int max_iter, float grad_tol,
                  int linesearch_max_iter, int lbfgs_memory, int verbosity,
                  float *w0, float *f, int *num_iters, bool X_col_major) {

  // TODO this will come from the cuml handle
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  cudaStream_t stream = 0;

  typedef float Real;
  typedef Softmax<Real> LossFunction;

  LossFunction loss(D, C, fit_intercept, cublas);
  SimpleMatOwning<Real> z(C, N); // TODO this allocation somewhere else?
  STORAGE_ORDER ord = X_col_major ? COL_MAJOR : ROW_MAJOR;

  qn_fit<Real, LossFunction>(loss, X, y, z.data, N, fit_intercept, l1, l2,
                             max_iter, grad_tol, linesearch_max_iter,
                             lbfgs_memory, verbosity, w0, f, num_iters, ord,
                             stream);
}

}; // namespace GLM
}; // namespace ML
