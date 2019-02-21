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
#include <glm/glm_vectors.h>
namespace ML {
namespace GLM {

enum LINE_SEARCH_ALGORITHM {
  LBFGS_LS_BT_ARMIJO = 1,
  LBFGS_LS_BT = 2, // Default. Alias for Wolfe
  LBFGS_LS_BT_WOLFE = 2,
  LBFGS_LS_BT_STRONG_WOLFE = 3
};

enum LINE_SEARCH_RETCODE {
  LS_SUCCESS = 0,
  LS_INVALID_STEP_MIN = 1,
  LS_INVALID_STEP_MAX = 2,
  LS_MAX_ITERS_REACHED = 3,
  LS_INVALID_DIR = 4,
  LS_INVALID_STEP = 5
};

enum OPT_RETCODE {
  OPT_SUCCESS = 0,
  OPT_NUMERIC_ERROR = 1,
  OPT_LS_FAILED = 2,
  OPT_MAX_ITERS_REACHED = 3,
  OPT_INVALID_ARGS = 4
};

template <typename T = double>
class LBFGSParam {
public:
  int m;     // lbfgs memory limit
  T epsilon; // controls convergence
  int past;  // lookback for function value based convergence test
  T delta;   // controls fun val based conv test
  int max_iterations;
  int linesearch; // see enum above
  int max_linesearch;
  T min_step; // min. allowed step length
  T max_step; // max. allowed step length
  T ftol;     // line  search tolerance
  T wolfe;    // wolfe parameter

public:
  LBFGSParam() {
    m = 6;
    epsilon = T(1e-5);
    past = 0;
    delta = T(0);
    max_iterations = 0;
    linesearch = LBFGS_LS_BT_ARMIJO;
    max_linesearch = 20;
    min_step = T(1e-20);
    max_step = T(1e+20);
    ftol = T(1e-4);
    wolfe = T(0.9);
  }

  inline int check_param() const { // TODO exceptions
    int ret = 1;
    if (m <= 0)
      return ret; 
    ret++;
    if (epsilon <= 0)
      return ret; 
    ret++;
    if (past < 0)
      return ret;
    ret++;
    if (delta < 0)
      return ret;
    ret++;
    if (max_iterations < 0)
      return ret; 
    ret++;
    if (linesearch < LBFGS_LS_BT_ARMIJO ||
        linesearch > LBFGS_LS_BT_STRONG_WOLFE)
      return ret; 
    ret++;
    if (max_linesearch <= 0)
      return ret;
    ret++;
    if (min_step < 0)
      return ret;
    ret++;
    if (max_step < min_step)
      return ret; 
    ret++;
    if (ftol <= 0 || ftol >= 0.5)
      return ret; 
    ret++;
    if (wolfe <= ftol || wolfe >= 1)
      return ret;
    ret++;
    return 0;
  }
};

template <typename T>
HDI T project_orth(T x, T y) {
  return x * y <= T(0) ? T(0) : x;
}

template <typename T>
HDI T get_pseudo_grad(T x, T dlossx, T C) {
  if (x != 0) {
    return dlossx + MLCommon::sgn(x) * C;
  }
  T dplus = dlossx + C;
  T dmins = dlossx - C;
  if (dmins > T(0))
    return dmins;
  if (dplus < T(0))
    return dplus;
  return T(0);
}

template <typename T>
struct op_project {
  T scal;
  op_project(T s) : scal(s) {}

  HDI T operator()(const T x, const T y) const {
    return project_orth(x, scal * y);
  }
};


template <typename T>
struct op_pseudo_grad {
  T l1;
  op_pseudo_grad(const T lam) : l1(lam) {}

  HDI T operator()(const T x, const T dlossx) const {
    return get_pseudo_grad(x, dlossx, l1);
  }
};


template <typename T>
struct LSProjectedStep {
  typedef SimpleVec<T> Vector;
  struct op_pstep {
    T step;
    op_pstep(const T s) : step(s) {}

    HDI T operator()(const T xp, const T drt, const T pg) const {
      T xi = xp == 0 ? -pg : xp;
      return project_orth(xp + step * drt, xi);
    }
  };

  void operator()(const T step, Vector &x, const Vector &drt, const Vector &xp,
                  const Vector &pgrad) const {
    op_pstep pstep(step);
    x.assign_ternary(xp, drt, pgrad, pstep);
  }
};

template <typename T>
struct LineSearch {
  typedef SimpleVec<T> Vector;
  T dec, inc;
  const LBFGSParam<T> &param;
  inner_product<T> dot;

  LineSearch(const LBFGSParam<T> &param_, T dec_ = 0.5, T inc_ = 2.1)
    : dec(dec_), inc(inc_), param(param_) {}

  bool is_success(const T fx_init, const T dg_init, const T fx, const T dg_test,
                  const T step, const Vector &grad, const Vector &drt,
                  T *width) {
    if (fx > fx_init + step * dg_test) {
      *width = dec;
    } else {
      // Armijo condition is met
      if (param.linesearch == LBFGS_LS_BT_ARMIJO)
        return true;

      const T dg = dot(grad, drt);
      if (dg < param.wolfe * dg_init) {
        *width = inc;
      } else {
        // Regular Wolfe condition is met
        if (param.linesearch == LBFGS_LS_BT_WOLFE)
          return true;

        if (dg > -param.wolfe * dg_init) {
          *width = dec;
        } else {
          // Strong Wolfe condition is met
          return true;
        }
      }
    }

    return false;
  }
  /*
   * Line search by backtracking.
   *
   * \param f      A function object such that `f(x, grad)` returns the
   *               objective function value at `x`, and overwrites `grad` with
   *               the gradient.
   * \param fx     In: The objective function value at the current point.
   *               Out: The function value at the new point.
   * \param x      Out: The new point moved to.
   * \param grad   In: The current gradient vector. Out: The gradient at the
   *               new point.
   * \param step   In: The initial step length. Out: The calculated step length.
   * \param drt    The current moving direction.
   * \param xp     The current point.
   */
  template <typename Function>
  LINE_SEARCH_RETCODE backtrack(Function &f, T &fx, Vector &x, Vector &grad,
                                T &step, const Vector &drt, const Vector &xp) {
    // Check the value of step
    if (step <= T(0))
      return LS_INVALID_STEP;

    // Save the function value at the current x
    const T fx_init = fx;
    // Projection of gradient on the search direction
    const T dg_init = dot(grad, drt);
    // Make sure d points to a descent direction
    if (dg_init > 0)
      return LS_INVALID_DIR;

    const T dg_test = param.ftol * dg_init;
    T width;

    int iter;
    for (iter = 0; iter < param.max_linesearch; iter++) {
      // x_{k+1} = x_k + step * d_k
      x.axpy(step, drt, xp);
      // Evaluate this candidate
      fx = f(x, grad);

      if (is_success(fx_init, dg_init, fx, dg_test, step, grad, drt, &width))
        return LS_SUCCESS;

      if (step < param.min_step)
        return LS_INVALID_STEP_MIN;

      if (step > param.max_step)
        return LS_INVALID_STEP_MAX;

      step *= width;
    }
    return LS_MAX_ITERS_REACHED;
  }

  template <typename Function>
  LINE_SEARCH_RETCODE
    backtrack_projected(Function &f, T &fx, Vector &x, Vector &grad,
                        const Vector &pseudo_grad, T &step, const Vector &drt,
                        const Vector &xp, T l1_penalty) {
    LSProjectedStep<T> lsstep;

    // Check the value of step
    if (step <= T(0))
      return LS_INVALID_STEP;

    // Save the function value at the current x
    const T fx_init = fx;
    // Projection of gradient on the search direction
    const T dg_init = dot(pseudo_grad, drt);
    // Make sure d points to a descent direction
    if (dg_init > 0)
      return LS_INVALID_DIR;

    const T dg_test = param.ftol * dg_init;
    T width;

    int iter;
    for (iter = 0; iter < param.max_linesearch; iter++) {
      // x_{k+1} = proj_orth(x_k + step * d_k)
      lsstep(step, x, drt, xp, pseudo_grad);
      fx =
        f(x, grad); // evaluates fx with l1 term, but only grad of the loss term

      if (is_success(fx_init, dg_init, fx, dg_test, step, pseudo_grad, drt,
                     &width))
        return LS_SUCCESS;

      if (step < param.min_step)
        return LS_INVALID_STEP_MIN;

      if (step > param.max_step)
        return LS_INVALID_STEP_MAX;

      step *= width;
    }
    return LS_MAX_ITERS_REACHED;
  }
};

template <typename T>
struct LBFGSSolver {
  typedef SimpleVec<T> Vector;
  typedef SimpleMat<T, COL_MAJOR> Matrix;

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

  LineSearch<T> linesearch;
  norm2<T> l2norm;
  inner_product<T> dot;

  /*
   * Ctor. Allocates workspace
   */
  LBFGSSolver(const LBFGSParam<T> &param, const int n)
    : m_param(param), m_s(n, param.m), m_y(n, param.m), m_xp(n), m_grad(n),
      m_gradp(n), m_drt(n), m_ys(param.m), m_alpha(param.m),
      m_fx(param.past > 0 ? param.past : 0), linesearch(param) {}

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
      m_alpha(param.m), m_fx(param.past > 0 ? param.past : 0) {}

  static int workspace_size(const LBFGSParam<T> &param, int n) {
    return (4 + 2 * param.m) * n;
  }

  template <typename Function>
  inline OPT_RETCODE minimize(Function &f, Vector &x, T &fx, int *k,
                              int verbosity = 0) {
    *k = 0;
    if (verbosity > 0) {
      printf("Running L-BFGS\n");
    }

    // Evaluate function and compute gradient
    fx = f(x, m_grad);
    T xnorm = l2norm(x);
    T gnorm = l2norm(m_grad);

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
    T step = T(1.0) / l2norm(m_drt);

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
      LINE_SEARCH_RETCODE lsret =
        linesearch.backtrack(f, fx, x, m_grad, step, m_drt, m_xp);
      if (lsret != LS_SUCCESS)
        return OPT_LS_FAILED;

      if (check_convergence(*k, fx, x, m_grad, verbosity)) {
        return OPT_SUCCESS;
      }

      // Update s and y
      update_state(end, x);
      // m_drt <- -H * g
      end = update_search_dir(*k, end, m_grad);
      // step = 1.0 as initial guess
      step = T(1.0);
    }
    return OPT_MAX_ITERS_REACHED;
  }

  bool check_convergence(const int k, const T fx, Vector &x, Vector &grad,
                         const int verbosity) {
    // New x norm and gradient norm
    T xnorm = l2norm(x);
    T gnorm = l2norm(grad);

    if (verbosity > 0) {
      printf("%04d: f(x)=%.6f conv.crit=%.6f (gnorm=%.6f, xnorm=%.6f)\n", k, fx,
             gnorm / std::max(T(1), xnorm), gnorm, xnorm);
    }
    // Convergence test -- gradient
    if (gnorm <= m_param.epsilon * std::max(xnorm, T(1.0))) {
      if (verbosity > 0)
        printf("Converged after %d iterations: f(x)=%.6f\n", k, fx);
      return true;
    }
    // Convergence test -- objective function value
    if (m_param.past > 0) {
      if (k >= m_param.past &&
          std::abs((m_fx[k % m_param.past] - fx) / fx) < m_param.delta) {
        if (verbosity > 0)
          printf("Insufficient change in objective value\n");
        return true;
      }

      m_fx[k % m_param.past] = fx;
    }
    return false;
  }

  /*
   * Updates memory
   * s_{k+1} = x_{k+1} - x_k
   * y_{k+1} = g_{k+1} - g_k
   */
  void update_state(const int end, const Vector &x) {
    col_ref(m_s,svec, end);
    col_ref(m_y,yvec, end);
    svec.axpy(-1.0, m_xp, x);
    yvec.axpy(-1.0, m_gradp, m_grad);
  }

  /*
   * Computes new search direction
   * d = - H * g,
   * where H is the approximate inverse Hessian
   */
  int update_search_dir(const int k, int end, const Vector &g) {
    // note: update_state assigned svec, yvec to m_s[:,end], m_y[:,end]
    T ys = dot(svec, yvec);
    T yy = dot(yvec, yvec);
    if (ys == 0 || yy == 0) {
      printf("WARNING: zero detected\n");
    }
    m_ys[end] = ys;

    // Recursive formula to compute d = -H * g
    m_drt.ax(-1.0, g);
    int bound = std::min(m_param.m, k);
    end = (end + 1) % m_param.m;
    int j = end;
    for (int i = 0; i < bound; i++) {
      j = (j + m_param.m - 1) % m_param.m;
      col_ref(m_s,sj, j);
      col_ref(m_y,yj, j);
      m_alpha[j] = dot(sj, m_drt) / m_ys[j];
      m_drt.axpy(-m_alpha[j], yj, m_drt);
    }

    m_drt.ax(ys / yy, m_drt);

    for (int i = 0; i < bound; i++) {
      col_ref(m_s,sj, j);
      col_ref(m_y,yj, j);
      T beta = dot(yj, m_drt) / m_ys[j];
      m_drt.axpy((m_alpha[j] - beta), sj, m_drt);
      j = (j + 1) % m_param.m;
    }

    return end;
  }
};

template <typename T>
struct OWLQNSolver : LBFGSSolver<T> {
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

  using Super::l2norm;
  using Super::linesearch;

  int pg_limit; //

  Vector m_pseudo;
  // op to project a vector onto the orthant of the neg of another vector
  op_project<T> project_neg;

  norm1<T> l1norm;

  OWLQNSolver(const LBFGSParam<T> &param, const int n, const int pg_limit)
      : Super(param, n), m_pseudo(n), project_neg(T(-1.0)) , pg_limit(pg_limit){
          ASSERT(pg_limit <= n, "OWL-QN: Invalid pseuda grad limit parameter");
      }

  OWLQNSolver(const LBFGSParam<T> &param, const int n, const int pg_limit, T *workspace)
    : Super(param, n, workspace), pg_limit(pg_limit),
      m_pseudo(n, workspace + Super::workspace_size(param, n)),
      project_neg(T(-1.0)) {
          ASSERT(pg_limit <= n, "OWL-QN: Invalid pseuda grad limit parameter");
      }

  static int workspace_size(const LBFGSParam<T> &param, int n) {
    return Super::workspace_size(param, n) + n;
  }

  inline void update_pseudo(const SimpleVec<T> & x, const op_pseudo_grad<T> & pseudo_grad){
      if(m_grad.len > pg_limit){
          m_pseudo = m_grad;
          SimpleVec<T> mask(m_pseudo.data, pg_limit);
          mask.assign_binary(x, m_grad, pseudo_grad);
      }else{
          m_pseudo.assign_binary(x, m_grad, pseudo_grad);
      }
  }

  template <typename Function>
  inline OPT_RETCODE minimize(Function &f, const T l1_penalty, Vector &x, T &fx,
                              int *k, const int verbosity = 0) {

    auto f_wrap = [&f, &l1_penalty, this](SimpleVec<T> &x, SimpleVec<T> &grad) {
      T tmp = f(x, grad);
      SimpleVec<T> mask(x.data, pg_limit);
      return tmp + l1_penalty * this->l1norm(mask);
    };

    *k = 0;
    if (verbosity > 0) {
      printf("Running OWL-QN with lambda=%f\n", l1_penalty);
    }

    // op to compute the pseudo gradients
    op_pseudo_grad<T> pseudo_grad(l1_penalty);

    fx = f_wrap(x, m_grad); // fx is loss+regularizer, grad is grad of loss only

    // compute pseudo grad, but don't overwrite grad: used to build H
    //m_pseudo.assign_binary(x, m_grad, pseudo_grad);
    update_pseudo(x, pseudo_grad);

    T xnorm = l2norm(x);
    T gnorm = l2norm(m_pseudo);

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
    T step = T(1.0) / std::max(T(1), l2norm(m_drt));

    int end = 0;
    for ((*k) = 1; (*k) <= m_param.max_iterations; (*k)++) {
      if (isnan(fx) || isinf(fx)) {
        return OPT_NUMERIC_ERROR;
      }
      // Save the curent x and gradient
      m_xp = x;
      m_gradp = m_grad;

      // Projected line search to update x, fx and gradient
      LINE_SEARCH_RETCODE lsret = linesearch.backtrack_projected(
        f_wrap, fx, x, m_grad, m_pseudo, step, m_drt, m_xp, l1_penalty);

      if (lsret != LS_SUCCESS)
        return OPT_LS_FAILED;
      // recompute pseudo
    //  m_pseudo.assign_binary(x, m_grad, pseudo_grad);
    update_pseudo(x, pseudo_grad);

      if (Super::check_convergence(*k, fx, x, m_pseudo, verbosity)) {
        return OPT_SUCCESS;
      }

      // Update s and y
      Super::update_state(end, x);
      // m_drt <- -H * g
      end = Super::update_search_dir(*k, end, m_pseudo);

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
inline int qn_minimize(SimpleVec<T> &x, T * fx, int * num_iters, LossFunction &loss, const T l1, const LBFGSParam<T> & opt_param,  const int verbosity = 0) {

  OPT_RETCODE ret;
  if (l1 == 0.0) {
    LBFGSSolver<T> lbfgs(opt_param, loss.n_param);
    ret = lbfgs.minimize(loss, x, *fx, num_iters, verbosity);

    if (verbosity > 0)
      printf("L-BFGS Done\n");
  } else {
//    opt_param.linesearch = LBFGS_LS_BT_ARMIJO; // Reference paper uses simple armijo ls...
    OWLQNSolver<T> owlqn(opt_param, loss.n_param, loss.D * loss.C);
    ret = owlqn.minimize(loss, l1, x, *fx, num_iters, verbosity);
    if (verbosity > 0)
      printf("OWL-QN Done\n");
  }
  return ret;
}

}; // namespace GLM
}; // namespace ML
