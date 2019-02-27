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
#include <glm/qn_solvers.h>

/*
 * Implementation of the quasi newton C api
 */

using namespace ML;
using namespace ML::GLM;

template <typename T, typename LossFunction>
int qn_fit(LossFunction *loss, T *Xptr, T *yptr, T *zptr, int N,
           bool fit_intercept, T l1, T l2, int max_iter, T grad_tol,
           T value_rel_tol, int linesearch_max_iter, int lbfgs_memory,
           int verbosity,
           T *w0, // initial value and result
           T *fx, int *num_iters, STORAGE_ORDER ordX) {

  LBFGSParam<T> opt_param;
  opt_param.epsilon = grad_tol;
  opt_param.delta = value_rel_tol;
  opt_param.max_iterations = max_iter;
  opt_param.m = lbfgs_memory;
  opt_param.max_linesearch = linesearch_max_iter;
  // opt_param.past = 1; //TODO if we wan delta to be used...
  SimpleVec<T> w(w0, loss->n_param);

  if (l2 == 0) {
    GLMWithData<T, LossFunction> lossWith(loss, Xptr, yptr, zptr, N, ordX);

    return qn_minimize(w, fx, num_iters, lossWith, l1, opt_param, verbosity);

  } else {

    Tikhonov<T> reg(l2);
    RegularizedGLM<T, LossFunction, decltype(reg)> obj(loss, &reg);
    GLMWithData<T, decltype(obj)> lossWith(&obj, Xptr, yptr, zptr, N, ordX);

    return qn_minimize(w, fx, num_iters, lossWith, l1, opt_param, verbosity);
  }
}

// TODO these could be macros..
void cuml_glm_logreg_fit_dqn(double *X, double *y, int N, int D,
                             bool fit_intercept, double l1, double l2,
                             int max_iter, double grad_tol,
                             double value_rel_tol, int linesearch_max_iter,
                             int lbfgs_memory, int verbosity,
                             double *w0, // initial value and result
                             double *f, int *num_iters, bool X_col_major) {

  typedef double Real;
  typedef LogisticLoss<Real> LossFunction;

  LossFunction loss(D, fit_intercept);
  SimpleVec<Real> z(N); // TODO this allocation somewhere else?

  if (X_col_major) {
    qn_fit<Real, LossFunction>(
        &loss, X, y, z.data, N, fit_intercept, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, COL_MAJOR);
  } else {
    qn_fit<Real, LossFunction>(
        &loss, X, y, z.data, N, fit_intercept, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, ROW_MAJOR);
  }
}

void cuml_glm_logreg_fit_sqn(float *X, float *y, int N, int D,
                             bool fit_intercept, float l1, float l2,
                             int max_iter, float grad_tol, float value_rel_tol,
                             int linesearch_max_iter, int lbfgs_memory,
                             int verbosity,
                             float *w0, // initial value and result
                             float *f, int *num_iters, bool X_col_major) {

  typedef float Real;
  typedef LogisticLoss<Real> LossFunction;

  LossFunction loss(D, fit_intercept);
  SimpleVec<Real> z(N); // TODO this allocation somewhere else?

  if (X_col_major) {
    qn_fit<Real, LossFunction>(
        &loss, X, y, z.data, N, fit_intercept, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, COL_MAJOR);
  } else {
    qn_fit<Real, LossFunction>(
        &loss, X, y, z.data, N, fit_intercept, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, ROW_MAJOR);
  }
}

void cuml_glm_linreg_fit_dqn(double *X, double *y, int N, int D,
                             bool fit_intercept, double l1, double l2,
                             int max_iter, double grad_tol,
                             double value_rel_tol, int linesearch_max_iter,
                             int lbfgs_memory, int verbosity,
                             double *w0, // initial value and result
                             double *f, int *num_iters, bool X_col_major) {
  typedef double Real;
  typedef SquaredLoss<Real> LossFunction;

  LossFunction loss(D, fit_intercept);
  SimpleVec<Real> z(N); // TODO this allocation somewhere else?

  if (X_col_major) {
    qn_fit<Real, LossFunction>(
        &loss, X, y, z.data, N, fit_intercept, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, COL_MAJOR);
  } else {
    qn_fit<Real, LossFunction>(
        &loss, X, y, z.data, N, fit_intercept, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, ROW_MAJOR);
  }
}

void cuml_glm_linreg_fit_sqn(float *X, float *y, int N, int D,
                             bool fit_intercept, float l1, float l2,
                             int max_iter, float grad_tol, float value_rel_tol,
                             int linesearch_max_iter, int lbfgs_memory,
                             int verbosity,
                             float *w0, // initial value and result
                             float *f, int *num_iters, bool X_col_major) {
  typedef float Real;
  typedef SquaredLoss<Real> LossFunction;

  LossFunction loss(D, fit_intercept);
  SimpleVec<Real> z(N); // TODO this allocation somewhere else?

  if (X_col_major) {
    qn_fit<Real, LossFunction>(
        &loss, X, y, z.data, N, fit_intercept, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, COL_MAJOR);
  } else {
    qn_fit<Real, LossFunction>(
        &loss, X, y, z.data, N, fit_intercept, l1, l2, max_iter, grad_tol,
        value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
        num_iters, ROW_MAJOR);
  }
}

void dummy(double *X, double *y, int N, int D, bool fit_intercept, double l1,
           double l2, int max_iter, double grad_tol, double value_rel_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity,
           double *w0, // initial value and result
           double *f, int *num_iters) {

  // need to instantiate templates, such that libcuml has them
  typedef double Real;

  LogisticLoss<Real> logistic(D, fit_intercept);
  Real *z = 0;

  qn_fit<Real, LogisticLoss<Real>>(
      &logistic, X, y, z, N, fit_intercept, l1, l2, max_iter, grad_tol,
      value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
      num_iters, ROW_MAJOR);

  Softmax<Real> softmax(1, 1, false);


  qn_fit<Real, Softmax<Real>>(
      &softmax, X, y, z, N, fit_intercept, l1, l2, max_iter, grad_tol,
      value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
      num_iters, COL_MAJOR);

  SquaredLoss<Real> squared(1, false);

  qn_fit<Real, SquaredLoss<Real>>(
      &squared, X, y, z, N, fit_intercept, l1, l2, max_iter, grad_tol,
      value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
      num_iters, ROW_MAJOR);

}

void dummy(float *X, float *y, int N, int D, bool fit_intercept, float l1,
           float l2, int max_iter, float grad_tol, float value_rel_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity,
           float *w0, // initial value and result
           float *f, int *num_iters) {

  // need to instantiate templates, such that libcuml has them
  typedef float Real;

  LogisticLoss<Real> logistic(D, fit_intercept);
  Real *z = 0;

  qn_fit<Real, LogisticLoss<Real>>(
      &logistic, X, y, z, N, fit_intercept, l1, l2, max_iter, grad_tol,
      value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
      num_iters, COL_MAJOR);


  Softmax<Real> softmax(1, 1, false);


  qn_fit<Real, Softmax<Real>>(
      &softmax, X, y, z, N, fit_intercept, l1, l2, max_iter, grad_tol,
      value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
      num_iters, COL_MAJOR);

  SquaredLoss<Real> squared(1, false);


  qn_fit<Real, SquaredLoss<Real>>(
      &squared, X, y, z, N, fit_intercept, l1, l2, max_iter, grad_tol,
      value_rel_tol, linesearch_max_iter, lbfgs_memory, verbosity, w0, f,
      num_iters, COL_MAJOR);
}
