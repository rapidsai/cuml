#include <cstdio>
#include <glm/glm_base.h>
#include <glm/glm_batch_gradient.h>
#include <glm/glm_logistic.h>
#include <glm/glm_softmax.h>
#include <glm/glm_regularizer.h>
#include <glm/lbfgs.h>
#include <glm/qn_c.h>

/*
 * Implementation of the quasi newton C api
 */

using namespace ML;
using namespace ML::GLM;


template <typename T, typename LossFunction, STORAGE_ORDER Storage = COL_MAJOR>
int qn_fit(LossFunction *loss, T *Xptr, T *yptr, T *zptr, int N, bool has_bias,
           T l1, T l2, int max_iter, T grad_tol, T value_rel_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity,
           T *w0, // initial value and result
           T *fx, int *num_iters) {

  LBFGSParam<T> opt_param;
  opt_param.epsilon = grad_tol;
  opt_param.delta = value_rel_tol;
  opt_param.max_iterations = max_iter;
  opt_param.m = lbfgs_memory;
  opt_param.max_linesearch = linesearch_max_iter;
  // opt_param.past = 1; //TODO if we wan delta to be used...
  SimpleVec<T> w(w0, loss->n_param);

  if (l2 == 0) {
    GLMWithData<T, LossFunction, Storage> lossWith(loss, Xptr, yptr, zptr,
                                                      N);

    return qn_minimize(w, fx, num_iters, lossWith, l1, opt_param, verbosity);

  } else {

    Tikhonov<T> reg(l2);
    RegularizedGLM<T, LossFunction, decltype(reg)> obj(loss, &reg);
    GLMWithData<T, decltype(obj), Storage> lossWith(&obj, Xptr, yptr, zptr, N);

    return qn_minimize(w, fx, num_iters, lossWith, l1, opt_param, verbosity);
  }
}

template <typename T, typename LossFunction>
int fit_dispatch(T *X, T *y, int N, int D, bool has_bias, T l1, T l2,
                 int max_iter, T grad_tol, T value_rel_tol,
                 int linesearch_max_iter, int lbfgs_memory, int verbosity,
                 T *w0, // initial value and result
                 T *fx, int *num_iters) {
  // scratch space to evaluate logits
  SimpleVec<T> eta(N);
  // differentiable part of the loss function
  LossFunction loss(X, y, eta.data, N, D, has_bias, l2);

  // wrap solution space
  SimpleVec<T> x(w0, loss.n_param);

  LBFGSParam<T> opt_param;
  opt_param.epsilon = grad_tol;
  opt_param.delta = value_rel_tol;
  opt_param.max_iterations = max_iter;
  opt_param.m = lbfgs_memory;
  opt_param.max_linesearch = linesearch_max_iter;
  // opt_param.past = 1; //TODO if we wan delta to be used...

  OPT_RETCODE ret;
  if (l1 == 0.0) {
    LBFGSSolver<T> lbfgs(opt_param, loss.n_param);
    ret = lbfgs.minimize(loss, x, *fx, num_iters, verbosity);

    if (verbosity > 0)
      printf("L-BFGS Done\n");
  } else {
    opt_param.linesearch =
        LBFGS_LS_BT_ARMIJO; // Reference paper uses simple armijo ls...
    OWLQNSolver<T> owlqn(opt_param, loss.n_param, D);
    ret = owlqn.minimize(loss, l1, x, *fx, num_iters, verbosity);
    if (verbosity > 0)
      printf("OWL-QN Done\n");
  }
  // TODO discuss return code convention
  return ret;
}

void cuml_glm_logreg_fit_dqn(double *X, double *y, int N, int D, bool has_bias,
                             double l1, double l2, int max_iter,
                             double grad_tol, double value_rel_tol,
                             int linesearch_max_iter, int lbfgs_memory,
                             int verbosity,
                             double *w0, // initial value and result
                             double *f, int *num_iters) {
  fit_dispatch<double, LogisticLoss<double>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);
}

void cuml_glm_logreg_fit_sqn(float *X, float *y, int N, int D, bool has_bias,
                             float l1, float l2, int max_iter, float grad_tol,
                             float value_rel_tol, int linesearch_max_iter,
                             int lbfgs_memory, int verbosity,
                             float *w0, // initial value and result
                             float *f, int *num_iters) {
  fit_dispatch<float, LogisticLoss<float>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);
}

void cuml_glm_linreg_fit_dqn(double *X, double *y, int N, int D, bool has_bias,
                             double l1, double l2, int max_iter,
                             double grad_tol, double value_rel_tol,
                             int linesearch_max_iter, int lbfgs_memory,
                             int verbosity,
                             double *w0, // initial value and result
                             double *f, int *num_iters) {
  fit_dispatch<double, SquaredLoss<double>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);
}

void cuml_glm_linreg_fit_sqn(float *X, float *y, int N, int D, bool has_bias,
                             float l1, float l2, int max_iter, float grad_tol,
                             float value_rel_tol, int linesearch_max_iter,
                             int lbfgs_memory, int verbosity,
                             float *w0, // initial value and result
                             float *f, int *num_iters) {
  fit_dispatch<float, SquaredLoss<float>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);
}

void dummy(double *X, double *y, int N, int D, bool has_bias, double l1,
           double l2, int max_iter, double grad_tol, double value_rel_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity,
           double *w0, // initial value and result
           double *f, int *num_iters) {

  fit_dispatch<double, LogisticLoss<double, ROW_MAJOR>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  fit_dispatch<double, LogisticLoss<double>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  fit_dispatch<double, SquaredLoss<double, ROW_MAJOR>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  fit_dispatch<double, SquaredLoss<double>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  LogisticLoss1<double> loss(D, has_bias);
  SimpleVec<double> z(N);
  qn_fit<double, LogisticLoss1<double>, COL_MAJOR>(
      &loss, X, y, z.data, N, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  qn_fit<double, LogisticLoss1<double>, ROW_MAJOR>(
      &loss, X, y, z.data, N, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  Softmax<double> test(1,1, false);

  qn_fit<double, Softmax<double>, ROW_MAJOR>(
      &test, X, y, z.data, N, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  qn_fit<double, Softmax<double>, COL_MAJOR>(
      &test, X, y, z.data, N, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

}

void dummy(float *X, float *y, int N, int D, bool has_bias, float l1, float l2,
           int max_iter, float grad_tol, float value_rel_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity,
           float *w0, // initial value and result
           float *f, int *num_iters) {

  fit_dispatch<float, LogisticLoss<float, ROW_MAJOR>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  fit_dispatch<float, LogisticLoss<float>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  fit_dispatch<float, SquaredLoss<float, ROW_MAJOR>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);

  fit_dispatch<float, SquaredLoss<float>>(
      X, y, N, D, has_bias, l1, l2, max_iter, grad_tol, value_rel_tol,
      linesearch_max_iter, lbfgs_memory, verbosity, w0, f, num_iters);
}
