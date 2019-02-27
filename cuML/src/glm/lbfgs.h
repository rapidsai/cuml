#pragma once

/*
 * This file contains implementations of two popular Quasi-Newton methods:
 * - Limited-memory Broyden Fletcher Goldfarb Shanno (L-BFGS) [Nocedal, Wright - Numerical Optimization (1999)]
 * - Orthant-wise limited-memory quasi-newton (OWL-QN) [Andrew, Gao - ICML 2007]
 *   https://www.microsoft.com/en-us/research/publication/scalable-training-of-l1-regularized-log-linear-models/
 *
 * L-BFGS is a classical method to solve unconstrained optimization problems of 
 * differentiable multi-variate functions f: R^D \mapsto R, i.e. it solves
 *
 * \min_{x \in R^D} f(x)
 *
 * iteratively by building up a m-dimensional (inverse) Hessian approximation.
 *
 * OWL-QN is an extension of L-BFGS that is specifically designed to optimize functions of the form
 *
 * f(x) + \lambda * \sum_i |x_i|,
 *
 * i.e. functions with an l1 penalty, by leveraging that |z| is differentiable when restricted to an orthant.
 *
 */

#include <glm/qn_util.h>
#include <glm/qn_linesearch.h>
#include <glm/glm_vectors.h>

namespace ML {
namespace GLM {

template <typename T> struct LBFGSSolver {
  typedef SimpleVec<T> Vector;
  typedef SimpleMat<T> Matrix;

  const LBFGSParam<T> &m_param; // Parameters to control the LBFGS algorithm
  Matrix m_s;                   // History of the s vectors
  Matrix m_y;                   // History of the y vectors
  Vector m_xp;                  // Old x
  Vector m_grad;                // New gradient
  Vector m_gradp;               // Old gradient
  Vector m_drt;                 // Moving direction
  std::vector<T> m_ys;          // History of the s'y values
  std::vector<T> m_alpha;       // History of the step lengths
  std::vector<T> m_fx;          // History of the objective function values
  Vector svec, yvec, yj, sj;    // helper mask vectors

  Vector dev_scalar; // device scalar for reduction results

  /*
   * Ctor. Allocates workspace
   */
  LBFGSSolver(const LBFGSParam<T> &param, const int n)
      : m_param(param), m_s(n, param.m), m_y(n, param.m), m_xp(n), m_grad(n),
        m_gradp(n), m_drt(n), m_ys(param.m), m_alpha(param.m),
        m_fx(param.past > 0 ? param.past : 0), dev_scalar(1) {}

  /*
   * Ctor with user-allocated workspace. Use workspace_size.
   * TODO: Data alignment??
   * TODO: inner_product/lpnorm ops allocate a scalar on device for holding
   * return values. The workspace doesnt contain those atm.
   */
  LBFGSSolver(const LBFGSParam<T> &param, const int n, T *workspace)
      : m_param(param), m_s(n, param.m, workspace),
        m_y(n, param.m, workspace + param.m * n),
        m_xp(n, workspace + 2 * param.m * n),
        m_grad(n, workspace + (2 * param.m + 1) * n),
        m_gradp(n, workspace + (2 * param.m + 2) * n),
        m_drt(n, workspace + (2 * param.m + 3) * n), m_ys(param.m),
        dev_scalar(1), m_alpha(param.m), m_fx(param.past > 0 ? param.past : 0) {
  }

  static int workspace_size(const LBFGSParam<T> &param, int n) {
    return (4 + 2 * param.m) * n + 1;
  }

  template <typename Function>
  inline OPT_RETCODE minimize(Function &f, Vector &x, T &fx, int *k,
                              int verbosity = 0, cudaStream_t stream = 0) {
    T *dptr = dev_scalar.data;
    *k = 0;
    if (verbosity > 0) {
      printf("Running L-BFGS\n");
    }

    // Evaluate function and compute gradient
    fx = f(x, m_grad);
    T xnorm = nrm2(x, dptr, stream);
    T gnorm = nrm2(m_grad, dptr, stream);

    if (m_param.past > 0)
      m_fx[0] = fx;

    // Early exit if the initial x is already a minimizer
    if (gnorm <= m_param.epsilon * std::max(xnorm, T(1.0))) {
      if (verbosity > 0) {
        printf("Initial solution fulfills optimality condition.\n");
      }
      return OPT_SUCCESS;
    }

    // Initial direction
    m_drt.ax(-1.0, m_grad);

    // Initial step
    T step = T(1.0) / nrm2(m_drt, dptr, stream);

    *k = 1;
    int end = 0;
    for (; *k <= m_param.max_iterations; (*k)++) {
      if (isnan(fx) || isinf(fx)) {
        return OPT_NUMERIC_ERROR;
      }
      // Save the curent x and gradient
      m_xp = x;
      m_gradp = m_grad;

      // Line search to update x, fx and gradient
      LINE_SEARCH_RETCODE lsret = ls_backtrack(m_param, f, fx, x, m_grad, step,
                                               m_drt, m_xp, dptr, stream);
      if (lsret != LS_SUCCESS)
        return OPT_LS_FAILED;

      if (check_convergence(m_param, *k, fx, x, m_grad, m_fx, verbosity, dptr,
                            stream)) {
        return OPT_SUCCESS;
      }

      // Update s and y
      update_state(end, x);
      // m_drt <- -H * g
      //
      ///end = update_search_dir(*k, end, m_grad);

      end = lbfgs_search_dir(m_param, *k,
                            end, m_s,
                            m_y, m_grad,
                            svec, yvec,
                            m_drt, m_ys,
                            m_alpha, dev_scalar.data,
                            stream) ;

      // step = 1.0 as initial guess
      step = T(1.0);
    }
    return OPT_MAX_ITERS_REACHED;
  }
  /*
   * Updates memory
   * s_{k+1} = x_{k+1} - x_k
   * y_{k+1} = g_{k+1} - g_k
   */
  void update_state(const int end, const Vector &x) {
    col_ref(m_s, svec, end);
    col_ref(m_y, yvec, end);
    svec.axpy(-1.0, m_xp, x);
    yvec.axpy(-1.0, m_gradp, m_grad);
  }

};

template <typename T> struct OWLQNSolver : LBFGSSolver<T> {
  typedef LBFGSSolver<T> Super;
  typedef SimpleVec<T> Vector;
  typedef SimpleMat<T> Matrix;

  using Super::m_alpha; // History of the step lengths
  using Super::m_drt;   // Moving direction
  using Super::m_fx;    // History of the objective function values
  using Super::m_grad;  // New gradient
  using Super::m_gradp; // Old gradient
  using Super::m_param; // Parameters to control the LBFGS algorithm
  using Super::m_s;     // History of the s vectors
  using Super::m_xp;    // Old x
  using Super::m_y;     // History of the y vectors
  using Super::m_ys;    // History of the s'y values

  // mask vectors
  using Super::sj;
  using Super::svec;
  using Super::yj;
  using Super::yvec;

  // using Super::l2norm;
  using Super::dev_scalar;

  int pg_limit; //

  Vector m_pseudo;
  // op to project a vector onto the orthant of the neg of another vector
  op_project<T> project_neg;

  // norm1<T> l1norm;

  OWLQNSolver(const LBFGSParam<T> &param, const int n, const int pg_limit)
      : Super(param, n), m_pseudo(n), project_neg(T(-1.0)), pg_limit(pg_limit) {
          //This parameter is a crude way to apply l1 penalty only to the first part
          //of the parameters, e.g. not to penalize the bias term in GLM,
          //as implemented in scikit learn
    ASSERT(pg_limit <= n, "OWL-QN: Invalid pseudo grad limit parameter");
  }

  OWLQNSolver(const LBFGSParam<T> &param, const int n, const int pg_limit,
              T *workspace)
      : Super(param, n, workspace), pg_limit(pg_limit),
        m_pseudo(n, workspace + Super::workspace_size(param, n)),
        project_neg(T(-1.0)) {
    ASSERT(pg_limit <= n, "OWL-QN: Invalid pseudo grad limit parameter");
  }

  static int workspace_size(const LBFGSParam<T> &param, int n) {
    return Super::workspace_size(param, n) + n;
  }

  inline void update_pseudo(const SimpleVec<T> &x,
                            const op_pseudo_grad<T> &pseudo_grad) {
    if (m_grad.len > pg_limit) {
      m_pseudo = m_grad;
      SimpleVec<T> mask(m_pseudo.data, pg_limit);
      mask.assign_binary(x, m_grad, pseudo_grad);
    } else {
      m_pseudo.assign_binary(x, m_grad, pseudo_grad);
    }
  }

  template <typename Function>
  inline OPT_RETCODE minimize(Function &f, const T l1_penalty, Vector &x, T &fx,
                              int *k, const int verbosity = 0,
                              cudaStream_t stream = 0) {

    T *dptr = dev_scalar.data;
    auto f_wrap = [&f, &l1_penalty, &dptr, this, &stream](SimpleVec<T> &x,
                                                 SimpleVec<T> &grad) {
      T tmp = f(x, grad);
      SimpleVec<T> mask(x.data, pg_limit);
      return tmp + l1_penalty * nrm1(mask, dptr, stream);
    };

    *k = 0;
    if (verbosity > 0) {
      printf("Running OWL-QN with lambda=%f\n", l1_penalty);
    }

    // op to compute the pseudo gradients
    op_pseudo_grad<T> pseudo_grad(l1_penalty);

    fx = f_wrap(x, m_grad); // fx is loss+regularizer, grad is grad of loss only

    // compute pseudo grad, but don't overwrite grad: used to build H
    // m_pseudo.assign_binary(x, m_grad, pseudo_grad);
    update_pseudo(x, pseudo_grad);

    T xnorm = nrm2(x, dptr, stream);
    T gnorm = nrm2(m_pseudo, dptr, stream);

    if (m_param.past > 0)
      m_fx[0] = fx;

    // Early exit if the initial x is already a minimizer
    if (gnorm <= m_param.epsilon * std::max(xnorm, T(1.0))) {
      if (verbosity > 0) {
        printf("Initial solution fulfills optimality condition.\n");
      }
      return OPT_SUCCESS;
    }

    // Initial direction
    m_drt.ax(-1.0, m_pseudo); // using Pseudo gradient here
    // below should be done for consistency but seems unnecessary
    // m_drt.assign_k_ary(project, m_pseudo, x);

    // Initial step
    T step = T(1.0) / std::max(T(1), nrm2(m_drt, dptr, stream));

    int end = 0;
    for ((*k) = 1; (*k) <= m_param.max_iterations; (*k)++) {
      if (isnan(fx) || isinf(fx)) {
        return OPT_NUMERIC_ERROR;
      }
      // Save the curent x and gradient
      m_xp = x;
      m_gradp = m_grad;

      // Projected line search to update x, fx and gradient
      LINE_SEARCH_RETCODE lsret =
          ls_backtrack_projected(m_param, f_wrap, fx, x, m_grad, m_pseudo, step,
                                 m_drt, m_xp, l1_penalty, dptr, stream);

      if (lsret != LS_SUCCESS)
        return OPT_LS_FAILED;
      // recompute pseudo
      //  m_pseudo.assign_binary(x, m_grad, pseudo_grad);
      update_pseudo(x, pseudo_grad);

      if (check_convergence(m_param, *k, fx, x, m_pseudo, m_fx, verbosity, dptr,
                            stream)) {
        return OPT_SUCCESS;
      }

      // Update s and y
      Super::update_state(end, x);
      // m_drt <- -H * -> pseudo grad <-
      end = lbfgs_search_dir(m_param, *k, end, m_s, m_y, m_pseudo, svec, yvec,
                             m_drt, m_ys, m_alpha, dev_scalar.data, stream);

      // Project m_drt onto orthant of -pseudog
      m_drt.assign_binary(m_drt, m_pseudo, project_neg);

      // step = 1.0 as initial guess
      step = T(1.0);
    }
    return OPT_MAX_ITERS_REACHED;
  }
};

/*
 * Chooses the right algorithm, depending on presence of l1 term
 */
template <typename T, typename LossFunction>
inline int qn_minimize(SimpleVec<T> &x, T *fx, int *num_iters,
                       LossFunction &loss, const T l1,
                       const LBFGSParam<T> &opt_param,
                       const int verbosity = 0) {

  OPT_RETCODE ret;
  if (l1 == 0.0) {
    LBFGSSolver<T> lbfgs(opt_param, loss.n_param);
    ret = lbfgs.minimize(loss, x, *fx, num_iters, verbosity);

    if (verbosity > 0)
      printf("L-BFGS Done\n");
  } else {
    //    opt_param.linesearch = LBFGS_LS_BT_ARMIJO; // Reference paper uses
    //    simple armijo ls...
    OWLQNSolver<T> owlqn(opt_param, loss.n_param, loss.D * loss.C);
    ret = owlqn.minimize(loss, l1, x, *fx, num_iters, verbosity);
    if (verbosity > 0)
      printf("OWL-QN Done\n");
  }
  return ret;
}

}; // namespace GLM
}; // namespace ML
