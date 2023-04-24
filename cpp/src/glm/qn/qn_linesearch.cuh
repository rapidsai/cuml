/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
 * Backtracking linesearch
 *
 * \param param        LBFGS parameters
 * \param f            A function object such that `f(x, grad)` returns the
 *                     objective function value at `x`, and overwrites `grad`
 *                     with the gradient.
 * \param fx           In: The objective function value at the current point.
 *                     Out: The function value at the new point.
 * \param x            Out: The new point moved to.
 * \param grad         In: The current gradient vector.
 *                     Out: The gradient at the new point.
 * \param step         In: The initial step length.
 *                     Out: The calculated step length.
 * \param drt          The current moving direction.
 * \param xp           The current point.
 * \param dev_scalar   Device pointer to workspace of at least 1
 * \param stream Device pointer to workspace of at least 1
 */
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
  // Check the value of step
  if (step <= T(0)) return LS_INVALID_STEP;

  // Save the function value at the current x
  const T fx_init = fx;
  // Projection of gradient on the search direction
  const T dg_init = dot(grad, drt, dev_scalar, stream);
  // Make sure d points to a descent direction
  if (dg_init > 0) return LS_INVALID_DIR;

  const T dg_test = param.ftol * dg_init;
  T width;

  CUML_LOG_TRACE("Starting line search fx_init=%f, dg_init=%f", fx_init, dg_init);

  int iter;
  for (iter = 0; iter < param.max_linesearch; iter++) {
    // x_{k+1} = x_k + step * d_k
    x.axpy(step, drt, xp, stream);
    // Evaluate this candidate
    fx = f(x, grad, dev_scalar, stream);
    CUML_LOG_TRACE("Line search iter %d, fx=%f", iter, fx);
    // if (is_success(fx_init, dg_init, fx, dg_test, step, grad, drt, &width))
    if (ls_success(
          param, fx_init, dg_init, fx, dg_test, step, grad, drt, &width, dev_scalar, stream))
      return LS_SUCCESS;

    if (step < param.min_step) return LS_INVALID_STEP_MIN;

    if (step > param.max_step) return LS_INVALID_STEP_MAX;

    step *= width;
  }
  return LS_MAX_ITERS_REACHED;
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
  LSProjectedStep<T> lsstep;

  // Check the value of step
  if (step <= T(0)) return LS_INVALID_STEP;

  // Save the function value at the current x
  const T fx_init = fx;
  // Projection of gradient on the search direction
  const T dg_init = dot(pseudo_grad, drt, dev_scalar, stream);
  // Make sure d points to a descent direction
  if (dg_init > 0) return LS_INVALID_DIR;

  const T dg_test = param.ftol * dg_init;
  T width;

  int iter;
  for (iter = 0; iter < param.max_linesearch; iter++) {
    // x_{k+1} = proj_orth(x_k + step * d_k)
    lsstep(step, x, drt, xp, pseudo_grad, stream);
    // evaluates fx with l1 term, but only grad of the loss term
    fx = f(x, grad, dev_scalar, stream);

    // if (is_success(fx_init, dg_init, fx, dg_test, step, pseudo_grad, drt,
    // &width))
    if (ls_success(
          param, fx_init, dg_init, fx, dg_test, step, pseudo_grad, drt, &width, dev_scalar, stream))
      return LS_SUCCESS;

    if (step < param.min_step) return LS_INVALID_STEP_MIN;

    if (step > param.max_step) return LS_INVALID_STEP_MAX;

    step *= width;
  }
  return LS_MAX_ITERS_REACHED;
}
};  // namespace detail
};  // namespace GLM
};  // namespace ML
