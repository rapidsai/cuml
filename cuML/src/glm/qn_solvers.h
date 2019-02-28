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

/*
 * This file contains implementations of two popular Quasi-Newton methods:
 * - Limited-memory Broyden Fletcher Goldfarb Shanno (L-BFGS) [Nocedal, Wright -
 * Numerical Optimization (1999)]
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
 * OWL-QN is an extension of L-BFGS that is specifically designed to optimize
 * functions of the form
 *
 * f(x) + \lambda * \sum_i |x_i|,
 *
 * i.e. functions with an l1 penalty, by leveraging that |z| is differentiable
 * when restricted to an orthant.
 *
 */

#include <cuda_utils.h>
#include <glm/glm_vectors.h>
#include <glm/qn_linesearch.h>
#include <glm/qn_util.h>

namespace ML {
namespace GLM {

using MLCommon::alignTo;

// TODO better way to deal with alignment? Smaller aligne possible?
constexpr size_t qn_align = 256;

template <typename T>
inline size_t lbfgs_workspace_size(const LBFGSParam<T> &param, const int n) {
  size_t mat_size = alignTo<size_t>(sizeof(T) * param.m * n, qn_align);
  size_t vec_size = alignTo<size_t>(sizeof(T) * n, qn_align);
  return 2 * mat_size + 4 * vec_size + qn_align;
}

template <typename T>
inline size_t owlqn_workspace_size(const LBFGSParam<T> &param, const int n) {
  size_t vec_size = alignTo<size_t>(sizeof(T) * n, qn_align);
  return lbfgs_workspace_size(param, n) + vec_size;
}

template <typename T, typename Function>
inline OPT_RETCODE min_lbfgs(const LBFGSParam<T> &param,
                             Function &f,     // function to minimize
                             SimpleVec<T> &x, // initial point, holds result
                             T &fx,           // output function value
                             int *k,          // output iterations
                             SimpleVec<T> &workspace, // scratch space
                             int verbosity = 0, cudaStream_t stream = 0) {

  int n = x.len;
  const int workspace_size = lbfgs_workspace_size(param, n);
  ASSERT(workspace.len >= workspace_size, "LBFGS: workspace insufficient");

  // SETUP WORKSPACE
  size_t mat_size = alignTo<size_t>(sizeof(T) * param.m * n, qn_align);
  size_t vec_size = alignTo<size_t>(sizeof(T) * n, qn_align);
  T *p_ws = workspace.data;
  SimpleMat<T> S(p_ws, n, param.m);
  p_ws += mat_size;
  SimpleMat<T> Y(p_ws, n, param.m);
  p_ws += mat_size;
  SimpleVec<T> xp(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> grad(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> gradp(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> drt(p_ws, n);
  p_ws += vec_size;
  T *dev_scalar = p_ws;

  SimpleVec<T> svec, yvec; // mask vectors

  std::vector<T> ys(param.m);
  std::vector<T> alpha(param.m);
  std::vector<T> fx_hist(param.past > 0 ? param.past : 0);

  *k = 0;

  if (verbosity > 0) {
    printf("Running L-BFGS\n");
  }

  // Evaluate function and compute gradient
  fx = f(x, grad, dev_scalar, stream);
  T xnorm = nrm2(x, dev_scalar, stream);
  T gnorm = nrm2(grad, dev_scalar, stream);

  if (param.past > 0)
    fx_hist[0] = fx;

  // Early exit if the initial x is already a minimizer
  if (gnorm <= param.epsilon * std::max(xnorm, T(1.0))) {
    if (verbosity > 0) {
      printf("Initial solution fulfills optimality condition.\n");
    }
    return OPT_SUCCESS;
  }

  // Initial direction
  drt.ax(-1.0, grad);

  // Initial step
  T step = T(1.0) / nrm2(drt, dev_scalar, stream);

  *k = 1;
  int end = 0;
  for (; *k <= param.max_iterations; (*k)++) {
    if (isnan(fx) || isinf(fx)) {
      return OPT_NUMERIC_ERROR;
    }
    // Save the curent x and gradient
    xp = x;
    gradp = grad;

    // Line search to update x, fx and gradient
    LINE_SEARCH_RETCODE lsret =
        ls_backtrack(param, f, fx, x, grad, step, drt, xp, dev_scalar, stream);
    if (lsret != LS_SUCCESS)
      return OPT_LS_FAILED;

    if (check_convergence(param, *k, fx, x, grad, fx_hist, verbosity,
                          dev_scalar, stream)) {
      return OPT_SUCCESS;
    }

    // Update s and y
    // s_{k+1} = x_{k+1} - x_k
    // y_{k+1} = g_{k+1} - g_k
    col_ref(S, svec, end);
    col_ref(Y, yvec, end);
    svec.axpy(-1.0, xp, x);
    yvec.axpy(-1.0, gradp, grad);
    // drt <- -H * g
    end = lbfgs_search_dir(param, *k, end, S, Y, grad, svec, yvec, drt, ys,
                           alpha, dev_scalar, stream);

    // step = 1.0 as initial guess
    step = T(1.0);
  }
  return OPT_MAX_ITERS_REACHED;
}

template <typename T>
inline void update_pseudo(const SimpleVec<T> &x, const SimpleVec<T> &grad,
                          const op_pseudo_grad<T> &pseudo_grad,
                          const int pg_limit, SimpleVec<T> &pseudo) {
  if (grad.len > pg_limit) {
    pseudo = grad;
    SimpleVec<T> mask(pseudo.data, pg_limit);
    mask.assign_binary(x, grad, pseudo_grad);
  } else {
    pseudo.assign_binary(x, grad, pseudo_grad);
  }
}

template <typename T, typename Function>
inline OPT_RETCODE min_owlqn(const LBFGSParam<T> &param, Function &f,
                             const T l1_penalty, const int pg_limit,
                             SimpleVec<T> &x, T &fx, int *k,
                             SimpleVec<T> &workspace, // scratch space
                             const int verbosity = 0, cudaStream_t stream = 0) {

  int n = x.len;
  const int workspace_size = owlqn_workspace_size(param, n);
  ASSERT(workspace.len >= workspace_size, "LBFGS: workspace insufficient");
  ASSERT(pg_limit <= n && pg_limit > 0,
         "OWL-QN: Invalid pseudo grad limit parameter");

  // SETUP WORKSPACE
  size_t mat_size = alignTo<size_t>(sizeof(T) * param.m * n, qn_align);
  size_t vec_size = alignTo<size_t>(sizeof(T) * n, qn_align);
  T *p_ws = workspace.data;
  SimpleMat<T> S(p_ws, n, param.m);
  p_ws += mat_size;
  SimpleMat<T> Y(p_ws, n, param.m);
  p_ws += mat_size;
  SimpleVec<T> xp(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> grad(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> gradp(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> drt(p_ws, n);
  p_ws += vec_size;
  SimpleVec<T> pseudo(p_ws, n);
  p_ws += vec_size;
  T *dev_scalar = p_ws;

  SimpleVec<T> svec, yvec; // mask vectors

  std::vector<T> ys(param.m);
  std::vector<T> alpha(param.m);
  std::vector<T> fx_hist(param.past > 0 ? param.past : 0);

  op_project<T> project_neg(T(-1.0));

  auto f_wrap = [&f, &l1_penalty, &pg_limit,
                 &stream](SimpleVec<T> &x, SimpleVec<T> &grad, T * dev_scalar, cudaStream_t stream = 0) {
    T tmp = f(x, grad, dev_scalar, stream);
    SimpleVec<T> mask(x.data, pg_limit);
    return tmp + l1_penalty * nrm1(mask, dev_scalar, stream);
  };

  *k = 0;
  if (verbosity > 0) {
    printf("Running OWL-QN with lambda=%f\n", l1_penalty);
  }

  // op to compute the pseudo gradients
  op_pseudo_grad<T> pseudo_grad(l1_penalty);

  fx = f_wrap(x, grad, dev_scalar, stream); // fx is loss+regularizer, grad is grad of loss only

  // compute pseudo grad, but don't overwrite grad: used to build H
  // pseudo.assign_binary(x, grad, pseudo_grad);
  update_pseudo(x, grad, pseudo_grad, pg_limit, pseudo);

  T xnorm = nrm2(x, dev_scalar, stream);
  T gnorm = nrm2(pseudo, dev_scalar, stream);

  if (param.past > 0)
    fx_hist[0] = fx;

  // Early exit if the initial x is already a minimizer
  if (gnorm <= param.epsilon * std::max(xnorm, T(1.0))) {
    if (verbosity > 0) {
      printf("Initial solution fulfills optimality condition.\n");
    }
    return OPT_SUCCESS;
  }

  // Initial direction
  drt.ax(-1.0, pseudo); // using Pseudo gradient here
  // below should be done for consistency but seems unnecessary
  // drt.assign_k_ary(project, pseudo, x);

  // Initial step
  T step = T(1.0) / std::max(T(1), nrm2(drt, dev_scalar, stream));

  int end = 0;
  for ((*k) = 1; (*k) <= param.max_iterations; (*k)++) {
    if (isnan(fx) || isinf(fx)) {
      return OPT_NUMERIC_ERROR;
    }
    // Save the curent x and gradient
    xp = x;
    gradp = grad;

    // Projected line search to update x, fx and gradient
    LINE_SEARCH_RETCODE lsret =
        ls_backtrack_projected(param, f_wrap, fx, x, grad, pseudo, step, drt,
                               xp, l1_penalty, dev_scalar, stream);

    if (lsret != LS_SUCCESS)
      return OPT_LS_FAILED;
    // recompute pseudo
    //  pseudo.assign_binary(x, grad, pseudo_grad);
    update_pseudo(x, grad, pseudo_grad, pg_limit, pseudo);

    if (check_convergence(param, *k, fx, x, pseudo, fx_hist, verbosity,
                          dev_scalar, stream)) {
      return OPT_SUCCESS;
    }

    // Update s and y
    col_ref(S, svec, end);
    col_ref(Y, yvec, end);
    svec.axpy(-1.0, xp, x);
    yvec.axpy(-1.0, gradp, grad);
    // drt <- -H * -> pseudo grad <-
    end = lbfgs_search_dir(param, *k, end, S, Y, pseudo, svec, yvec, drt, ys,
                           alpha, dev_scalar, stream);

    // Project drt onto orthant of -pseudog
    drt.assign_binary(drt, pseudo, project_neg);

    // step = 1.0 as initial guess
    step = T(1.0);
  }
  return OPT_MAX_ITERS_REACHED;
}
/*
 * Chooses the right algorithm, depending on presence of l1 term
 */
template <typename T, typename LossFunction>
inline int qn_minimize(SimpleVec<T> &x, T *fx, int *num_iters,
                       LossFunction &loss, const T l1,
                       const LBFGSParam<T> &opt_param,
                       const int verbosity = 0) {

  // TODO should the worksapce allocation happen outside?
  OPT_RETCODE ret;
  if (l1 == 0.0) {
    SimpleVecOwning<T> workspace(lbfgs_workspace_size(opt_param, x.len));

    ret = min_lbfgs(opt_param,
                    loss,      // function to minimize
                    x,         // initial point, holds result
                    *fx,       // output function value
                    num_iters, // output iterations
                    workspace, // scratch space
                    verbosity);

    if (verbosity > 0)
      printf("L-BFGS Done\n");
  } else { 
      // There might not be a better way to deal with dispatching
      // for the l1 case:
      // The algorithm explicitely expects a differentiable 
      // function f(x). It takes care of adding and
      // handling the term l1norm(x) * l1_pen explicitely, i.e.
      // it needs to evaluate f(x) and its gradient separately

    SimpleVecOwning<T> workspace(owlqn_workspace_size(opt_param, x.len));

    ret = min_owlqn(opt_param,
                    loss, // function to minimize
                    l1, loss.D * loss.C,
                    x,         // initial point, holds result
                    *fx,       // output function value
                    num_iters, // output iterations
                    workspace, // scratch space
                    verbosity);

    if (verbosity > 0)
      printf("OWL-QN Done\n");
  }
  return ret;
}

}; // namespace GLM
}; // namespace ML
