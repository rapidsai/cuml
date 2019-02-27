#pragma once

namespace ML{
namespace GLM{



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
  T ls_dec; //line search decrease factor
  T ls_inc; //line search increase factor

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
    ls_dec = T(0.5);
    ls_inc = T(2.1);
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
inline bool check_convergence(const LBFGSParam<T> &param, const int k, const T fx,
                       SimpleVec<T> &x, SimpleVec<T> &grad, std::vector<T> & fx_hist, const int verbosity,
                       T *dev_scalar, cudaStream_t stream = 0) {
  // New x norm and gradient norm
  T xnorm = nrm2(x, dev_scalar, stream);
  T gnorm = nrm2(grad, dev_scalar, stream);

  if (verbosity > 0) {
    printf("%04d: f(x)=%.6f conv.crit=%.6f (gnorm=%.6f, xnorm=%.6f)\n", k, fx,
           gnorm / std::max(T(1), xnorm), gnorm, xnorm);
  }
  // Convergence test -- gradient
  if (gnorm <= param.epsilon * std::max(xnorm, T(1.0))) {
    if (verbosity > 0)
      printf("Converged after %d iterations: f(x)=%.6f\n", k, fx);
    return true;
  }
  // Convergence test -- objective function value
  if (param.past > 0) {
    if (k >= param.past &&
        std::abs((fx_hist[k % param.past] - fx) / fx) < param.delta) {
      if (verbosity > 0)
        printf("Insufficient change in objective value\n");
      return true;
    }

    fx_hist[k % param.past] = fx;
  }
  return false;
}

}; // namespace GLM
}; // namespace ML
