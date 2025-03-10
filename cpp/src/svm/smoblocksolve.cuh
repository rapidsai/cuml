/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

/**@file smoblocksolve.cuh  contains implementation of the blocke SMO solver
 */
#pragma once

#include "smo_sets.cuh"

#include <cuml/common/functional.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/svm/svm_parameter.h>

#include <raft/util/cuda_utils.cuh>

#include <selection/kselection.cuh>
#include <stdlib.h>

namespace ML {
namespace SVM {

/**
 * @brief Solve the optimization problem for the actual working set.
 *
 * Based on Platt's SMO [1], using improvements from Keerthy and Shevade [2].
 * A concise summary of the math can be found in Appendix A1 of [3].
 * We solve the QP subproblem for the vectors in the working set (WS).
 *
 * Let us first discuss classification (C-SVC):
 *
 * We would like to maximize the following quantity
 * \f[ W(\mathbf{\alpha}) = -\mathbf{\alpha}^T \mathbf{1}
 *   + \frac{1}{2} \mathbf{\alpha}^T Q \mathbf{\alpha}, \f]
 * subject to
 * \f[ \mathbf{\alpha}^T \mathbf{y} = 0 \\
 *     \mathbf{0} \le \mathbf{\alpha}\le C \mathbf{1},\f]
 * where \f$ Q_{i,j} = y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)\f$
 *
 * This corresponds to Lagrangian for the dual is:
 * \f[ L = \frac{1}{2}  \sum_{i,j}\alpha_i Q \alpha_j - \sum_i \alpha_i
 *      -\sum_i \delta_i \alpha_i + \sum_i\mu_i(\alpha_i -C)
 *      - \beta \sum_i \alpha_i y_i
 *\f]
 *
 * Let us define the optimality indicator vector
 * \f[ f_i = y_i
 *     \frac{\partial W(\mathbf{\alpha})}{\partial \alpha_i} =
 *     -y_i +   y_j \alpha_j K(\mathbf{x}_i, \mathbf{x}_j) =
 *     -y_i + y_i Q_{i,j} \alpha_j.
 * \f]

 * The Karush-Kuhn-Tucker conditions are necessary and sufficient for optimality.
 * According to [2], the conditions simplify to
 * \f[ \beta \le f_i, \forall i \in I_\mathrm{upper}, \quad
 *     \beta \ge f_i \forall i \in I_\mathrm{lower}. \f]
 *
 * If \f$ \max\{f_i | i \in I_\mathrm{lower}\} \le \min\{f_i| i\in I_\mathrm{upper}\}\f$,
 * then we are converged because any beta value in this interval would lead to
 * an optimal solution. Otherwise we modify the alpha parameters until the
 * corresponding changes in f lead to an on optimal solution.
 *
 * Before the first iteration, one should set \f$ \alpha_i = 0\f$, and
 * \f$ f_i = -y_i \f$, for each \f$ i \in [0..n_{rows}]\f$.
 *
 * To find the optimal alpha parameters, we use the SMO method: we select two
 * vectors u and l from the WS and update the dual coefficients of these vectors.
 * We iterate several times, and accumulate the change in the dual coeffs in
 * \f$\Delta\alpha\f$.
 *
 * In every iteration we select the two vectors using the following formulas
 *   \f[ u = \mathrm{argmin}_{i=1}^{n_{ws}}\left[ f_i |
 *      x_i \in X_\mathrm{upper} \right] \f]
 *
 *  \f[ l = \mathrm{argmax}_{i=1}^{n_{ws}} \left[
 *          \frac{(f_u-f_i)^2}{\eta_i}| f_u < f_i \land x_i \in
 *           X_{\mathrm{lower}}\right], \f]
 *  where \f[\eta_i = K(x_u, x_u) + K(x_i, x_i) - 2K(x_u, x_i). \f]
 *
 * We update the values of the dual coefs according to (additionally we clip
 * values so that the coefficients stay in the [0, C] interval)
 * \f[ \Delta \alpha_l = - y_l \frac{f_l - f_u}{\eta_l} = -y_l q, \f]
 * \f[ \alpha_l += \Delta \alpha_l, \f]
 * \f[ \Delta \alpha_u = -y_u y_l \Delta \alpha_l = y_u q, \f]
 * \f[ \alpha_u += \Delta \alpha_u. \f]
 *
 * We also update the optimality indicator vector for the WS:
 * \f[ f_i += \Delta\alpha_u y_u K(x_u,x_i) + \Delta\alpha_l y_l K(x_l, x_i) \f]
 *
 * During the inner iterations, the f values are updated only for the WS
 * (since we are solving the optimization subproblem for the WS subproblem).
 * For consistency, f is kept as an input parameter, the changed values are
 * not saved back to global memory. After this solver finishes,  all the f
 * values (WS and outside WS) f should be updated using the delta_alpha output
 * parameter.
 *
 * For SVR, we do the same steps to solve the problem. The difference is the
 * optimization objective (W), which enters only as the initial value of f:
 *
 * \f[
 * W(\alpha^+, \alpha^-) =
 * \epsilon \sum_{i=1}^l (\alpha_i^+ + \alpha_i^-)
 * - \sum_{i=1}^l yc_i (\alpha_i^+ - \alpha_i^-)
 * + \frac{1}{2} \sum_{i,j=1}^l
 *   (\alpha_i^+ - \alpha_i^-)(\alpha_j^+ - \alpha_j^-) K(\bm{x}_i, \bm{x}_j)
 * \f]
 *
 * References:
 * - [1] J. C. Platt Sequential Minimal Optimization: A Fast Algorithm for
 *      Training Support Vector Machines, Technical Report MS-TR-98-14 (1998)
 * - [2] S.S. Keerthi et al. Improvements to Platt's SMO Algorithm for SVM
 *      Classifier Design, Neural Computation 13, 637-649 (2001)
 * - [3] Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs, Journal
 *      of Machine Learning Research, 19, 1-5 (2018)
 *
 * @tparam math_t floating point data type
 * @tparam WSIZE working set size (max 1024)
 * @param [in] y_array target labels size [n_train]
 * @param [in] n_train number of training vectors
 * @param [inout] alpha dual coefficients, size [n_train]
 * @param [in] n_ws number of elements in the working set
 * @param [out] delta_alpha change in the dual coeff of vectors in the working
 *        set, size [n_ws]
 * @param [in] f_array optimality indicator vector, size [n_train]
 * @param [in] kernel kernel function calculated between the working set and all
 *   other training vectors, size [n_ws * n_ws]
 * @param [in] ws_idx indices of traning vectors in the working set, size [n_ws]
 * @param [in] C_vec penalty parameter vector including class and sample weights
 *   size [n_train]
 * @param [in] eps tolerance, iterations will stop if the duality gap is smaller
 *  than this value (or if the gap is smaller than 0.1 times the initial gap)
 * @param [out] return_buff, two values are returned: duality gap and the number
 *   of iterations
 * @param [in] max_iter maximum number of iterations
 * @param [in] svmType type of the SVM problem to solve
 */
template <typename math_t, int WSIZE>
CUML_KERNEL __launch_bounds__(WSIZE) void SmoBlockSolve(math_t* y_array,
                                                        int n_train,
                                                        math_t* alpha,
                                                        int n_ws,
                                                        math_t* delta_alpha,
                                                        math_t* f_array,
                                                        const math_t* kernel,
                                                        const int* ws_idx,
                                                        const math_t* C_vec,
                                                        math_t eps,
                                                        math_t* return_buff,
                                                        int max_iter    = 10000,
                                                        SvmType svmType = C_SVC)
{
  typedef MLCommon::Selection::KVPair<math_t, int> Pair;
  typedef cub::BlockReduce<Pair, WSIZE> BlockReduce;
  typedef cub::BlockReduce<math_t, WSIZE> BlockReduceFloat;
  __shared__ union {
    typename BlockReduce::TempStorage pair;
    typename BlockReduceFloat::TempStorage single;
  } temp_storage;

  // From Platt [1]: "Under unusual circumstances \eta will not be positive.
  // A negative \eta will occur if the kernel K does note obey Mercer's
  // condition [...]. A zero \eta can occur even with a correct kernel, if more
  // than one training example has the input vector x." We set a lower limit to
  // \eta, to ensure correct behavior of SMO.
  constexpr math_t ETA_EPS = 1.0e-12;  // minimum value for \eta

  __shared__ math_t f_u;
  __shared__ int u;
  __shared__ int l;

  __shared__ math_t tmp_u, tmp_l;
  __shared__ math_t Kd[WSIZE];  // diagonal elements of the kernel matrix

  int tid = threadIdx.x;
  int idx = ws_idx[tid];

  // store values in registers
  math_t y      = y_array[idx];
  math_t f      = f_array[idx];
  math_t a      = alpha[idx];
  math_t a_save = a;
  math_t C      = C_vec[idx];

  __shared__ math_t diff_end;
  __shared__ math_t diff;

  Kd[tid]    = kernel[tid + tid * n_ws];
  int n_iter = 0;

  for (; n_iter < max_iter; n_iter++) {
    // mask values outside of X_upper
    math_t f_tmp = in_upper(a, y, C) ? f : INFINITY;
    Pair pair{f_tmp, tid};
    Pair res = BlockReduce(temp_storage.pair).Reduce(pair, ML::detail::minimum{}, n_ws);
    if (tid == 0) {
      f_u = res.val;
      u   = res.key;
    }
    // select f_max to check stopping condition
    f_tmp = in_lower(a, y, C) ? f : -INFINITY;
    __syncthreads();  // needed because we are reusing the shared memory buffer
                      // and also the u shared value
    math_t Kui   = kernel[u * n_ws + tid];
    math_t f_max = BlockReduceFloat(temp_storage.single).Reduce(f_tmp, ML::detail::maximum{}, n_ws);

    if (tid == 0) {
      // f_max-f_u is used to check stopping condition.
      diff = f_max - f_u;
      if (n_iter == 0) {
        return_buff[0] = diff;
        diff_end       = max(eps, 0.1f * diff);
      }
    }
    __syncthreads();
    if (diff < diff_end) { break; }

    if (f_u < f && in_lower(a, y, C)) {
      math_t eta_ui = max(Kd[tid] + Kd[u] - 2 * Kui, ETA_EPS);
      f_tmp         = (f_u - f) * (f_u - f) / eta_ui;
    } else {
      f_tmp = -INFINITY;
    }
    pair = Pair{f_tmp, tid};
    res  = BlockReduce(temp_storage.pair).Reduce(pair, ML::detail::maximum{}, n_ws);
    if (tid == 0) { l = res.key; }
    __syncthreads();
    math_t Kli = kernel[l * n_ws + tid];

    // Update alpha
    // Let's set q = \frac{f_l - f_u}{\eta_{ul}
    // Ideally we would have a'_u = a_u + y_u*q and a'_l = a_l - y_l*q
    // We know that 0 <= a <= C, and the updated values (a') should also stay in
    // this range. Therefore
    // 0 <= a_u + y_u *q <= C   -->   -a_u <= y_u * q <= C - a_u
    // Based on the value of y_u we have two branches:
    // y == 1: -a_u <= q <= C-a_u and y == -1: a_u >= q >= a_u - C
    // Knowing that q > 0 (since f_l > f_u and \eta_ul > 0), and 0 <= a_u <= C,
    // the constraints are simplified as
    // y == 1:  q <= C-a_u, and  y == -1: q <= a_u
    // Similarly we can say for a'_l:
    // y == 1:  q <= a_l, and y ==- 1: q <= C - a_l
    // We clip q accordingly before we do the update of a.
    if (threadIdx.x == u) tmp_u = y > 0 ? C - a : a;
    if (threadIdx.x == l) {
      tmp_l = y > 0 ? a : C - a;
      // note: Kui == Kul for this thread
      math_t eta_ul = max(Kd[u] + Kd[l] - 2 * Kui, ETA_EPS);

      tmp_l = min(tmp_l, (f - f_u) / eta_ul);
    }
    __syncthreads();
    math_t q = min(tmp_u, tmp_l);
    if (threadIdx.x == u) a += q * y;
    if (threadIdx.x == l) a -= q * y;
    f += q * (Kui - Kli);
    if (q == 0) {
      // Probably fp underflow
      break;
    }
  }
  // save results to global memory before exit
  alpha[idx] = a;
  // it is actually y * \Delta \alpha

  // This is equivalent with: delta_alpha[tid] = (a - a_save) * y;
  delta_alpha[tid] = (a - a_save) * y;

  // f is recalculated in f_update, therefore we do not need to save that
  return_buff[1] = n_iter;
}
};  // end namespace SVM
};  // end namespace ML
