/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "qn_util.cuh"

/*
 * Linesearch functions
 */

namespace ML {
namespace GLM {
namespace detail {

template <typename T>
struct LSProjectedStep {
  typedef SimpleVec<T> Vector;
  struct op_pstep {
    T step;
    op_pstep(const T s) : step(s) {}

    HDI T operator()(const T xp, const T drt, const T pg) const
    {
      T xi = xp == 0 ? -pg : xp;
      return project_orth(xp + step * drt, xi);
    }
  };

  void operator()(const T step,
                  Vector& x,
                  const Vector& drt,
                  const Vector& xp,
                  const Vector& pgrad,
                  cudaStream_t stream) const
  {
    op_pstep pstep(step);
    x.assign_ternary(xp, drt, pgrad, pstep, stream);
  }
};

template <typename T>
struct LSNonnegStep {
  typedef SimpleVec<T> Vector;
  struct op_nonneg_step {
    T step;
    op_nonneg_step(const T s) : step(s) {}

    HDI T operator()(const T xp, const T drt) const { return raft::max(xp + step * drt, T(0)); }
  };

  void operator()(
    const T step, Vector& x, const Vector& drt, const Vector& xp, cudaStream_t stream) const
  {
    op_nonneg_step nstep(step);
    x.assign_binary(xp, drt, nstep, stream);
  }
};

template <typename T>
inline bool ls_success(const LBFGSParam<T>& param,
                       const T fx_init,
                       const T dg_init,
                       const T fx,
                       const T dg_test,
                       const T step,
                       const SimpleVec<T>& grad,
                       const SimpleVec<T>& drt,
                       T* width,
                       T* dev_scalar,
                       cudaStream_t stream)
{
  if (fx > fx_init + step * dg_test) {
    *width = param.ls_dec;
  } else {
    // Armijo condition is met
    if (param.linesearch == LBFGS_LS_BT_ARMIJO) return true;

    const T dg = dot(grad, drt, dev_scalar, stream);
    if (dg < param.wolfe * dg_init) {
      *width = param.ls_inc;
    } else {
      // Regular Wolfe condition is met
      if (param.linesearch == LBFGS_LS_BT_WOLFE) return true;

      if (dg > -param.wolfe * dg_init) {
        *width = param.ls_dec;
      } else {
        // Strong Wolfe condition is met
        return true;
      }
    }
  }

  return false;
}

/**
 * Common backtracking linesearch loop.
 *
 * @param dg_grad   The gradient vector used for computing the directional derivative
 *                  and for the Wolfe condition check (grad for standard/nonneg, pseudo_grad
 *                  for OWL-QN projected variant).
 * @param step_fn   Callable `(T step) -> void` that updates x from xp along drt.
 */
template <typename T, typename Function, typename StepFn>
LINE_SEARCH_RETCODE ls_backtrack_impl(const LBFGSParam<T>& param,
                                      Function& f,
                                      T& fx,
                                      SimpleVec<T>& x,
                                      SimpleVec<T>& grad,
                                      const SimpleVec<T>& dg_grad,
                                      T& step,
                                      const SimpleVec<T>& drt,
                                      T* dev_scalar,
                                      cudaStream_t stream,
                                      StepFn step_fn)
{
  if (step <= T(0)) return LS_INVALID_STEP;

  const T fx_init = fx;
  const T dg_init = dot(dg_grad, drt, dev_scalar, stream);
  if (dg_init > 0) return LS_INVALID_DIR;

  const T dg_test = param.ftol * dg_init;
  T width;

  CUML_LOG_TRACE("Starting line search fx_init=%f, dg_init=%f", fx_init, dg_init);

  for (int iter = 0; iter < param.max_linesearch; iter++) {
    step_fn(step);
    fx = f(x, grad, dev_scalar, stream);
    CUML_LOG_TRACE("Line search iter %d, fx=%f", iter, fx);

    if (ls_success(
          param, fx_init, dg_init, fx, dg_test, step, dg_grad, drt, &width, dev_scalar, stream))
      return LS_SUCCESS;

    if (step < param.min_step) return LS_INVALID_STEP_MIN;
    if (step > param.max_step) return LS_INVALID_STEP_MAX;

    step *= width;
  }
  return LS_MAX_ITERS_REACHED;
}

template <typename T, typename Function>
LINE_SEARCH_RETCODE ls_backtrack(const LBFGSParam<T>& param,
                                 Function& f,
                                 T& fx,
                                 SimpleVec<T>& x,
                                 SimpleVec<T>& grad,
                                 T& step,
                                 const SimpleVec<T>& drt,
                                 const SimpleVec<T>& xp,
                                 T* dev_scalar,
                                 cudaStream_t stream)
{
  return ls_backtrack_impl(param, f, fx, x, grad, grad, step, drt, dev_scalar, stream, [&](T s) {
    x.axpy(s, drt, xp, stream);
  });
}

template <typename T, typename Function>
LINE_SEARCH_RETCODE ls_backtrack_projected(const LBFGSParam<T>& param,
                                           Function& f,
                                           T& fx,
                                           SimpleVec<T>& x,
                                           SimpleVec<T>& grad,
                                           const SimpleVec<T>& pseudo_grad,
                                           T& step,
                                           const SimpleVec<T>& drt,
                                           const SimpleVec<T>& xp,
                                           T l1_penalty,
                                           T* dev_scalar,
                                           cudaStream_t stream)
{
  return ls_backtrack_impl(
    param, f, fx, x, grad, pseudo_grad, step, drt, dev_scalar, stream, [&](T s) {
      LSProjectedStep<T>{}(s, x, drt, xp, pseudo_grad, stream);
    });
}

template <typename T, typename Function>
LINE_SEARCH_RETCODE ls_backtrack_nonneg(const LBFGSParam<T>& param,
                                        Function& f,
                                        T& fx,
                                        SimpleVec<T>& x,
                                        SimpleVec<T>& grad,
                                        T& step,
                                        const SimpleVec<T>& drt,
                                        const SimpleVec<T>& xp,
                                        T* dev_scalar,
                                        cudaStream_t stream)
{
  return ls_backtrack_impl(param, f, fx, x, grad, grad, step, drt, dev_scalar, stream, [&](T s) {
    LSNonnegStep<T>{}(s, x, drt, xp, stream);
  });
}

};  // namespace detail
};  // namespace GLM
};  // namespace ML
