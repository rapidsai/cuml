/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <stdlib.h>
#include "ml_utils.h"
#include "selection/kselection.h"
#include "smo_sets.h"
namespace ML {
namespace SVM {

/**
 * Solve the optimization problem for the actual working set.
 *
 * Based on Platt's SMO [1], using improvements from Keerthy and Shevade [2].
 * A concise summary of the math can be found in Appendix A1 of [3].
 *
 * Before the first iteration, one should set \f$ \alpha_i = 0\f$, and
 * \f$ f_i = -y_i \f$, for each \f$ i \in [0..n_{rows}]\f$.
 *
 * We solve the QP subproblem for the vectors in the working set (WS).
 * We use the SMO method: we select two vectors u and l from the WS and update
 * the dual cofficients of these vectors. We iterate several times, and
 * accumulate the change in the dual coeffs in \f$\Delta\alpha\f$.
 *
 * In every iteration we select the two vectors using the following formulas
 *   \f[ u = \mathrm{argmax}_{i=1}^{n_{ws}}\left[ f_i |
 *      x_i \in X_\mathrm{upper} \right] \f]
 *
 *  \f[ l = \mathrm{argmax}_{i=1}^n_{ws}} \left[
 *          \frac{(f_u-f_i)^2}{\eta_i}| f_u < f_i \and x_i \in
 *           X_{\mathrm{lower}}\right], \f]
 *  where \f[\eta_i = K(x_u, x_u) + K(x_i, x_i) - 2K(x_u, x_i). \f]
 *
 * We update the values of the dual coefs according to (additionaly we clip
 * values so that the coefficients stay in the [0, C] interval)
 * \f[ \Delta \alpha_l = y_l \frac{f_u - f_l}{\eta_l}, \f]
 * \f[ \alpha_l += \Delta \alpha_l, \f]
 * \f[ \Delta \alpha_u = -y_u y_l \Delta \alpha_l, \f]
 * \f[ \alpha_l += \Delta \alpha_l. \f]
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
 * References:
 *  [1] J. C. Platt Sequential Minimal Optimization: A Fast Algorithm for
 *      Training Support Vector Machines, Technical Report MS-TR-98-14 (1998)
 *  [2] S.S. Keerthi et al. Improvements to Platt's SMO Algorithm for SVM
 *      Classifier Design, Neural Computation 13, 637-649 (2001)
 *  [3] Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs, Journal
 *      of Machine Learning Research, 19, 1-5 (2018)
 *
 * @tparam math_t floating point data type
 * @tparam WSIZE working set size (max 1024, should be divisible by 4)
 * @param [in] y_array target labels size [n_rows]
 * @param [in] n_rows number of trainig vectors
 * @param [inout] alpha dual coefficients, size [n_rows]
 * @param [in] n_ws number of elements in the working set
 * @param [out] delta_alpha change in the dual coeff of vectors in the working
 *        set, size [n_ws]
 * @param [in] f_array optimality indicator vector, size [n_rows]
 * @param [in] kernel kernel function calculated between the working set and all
 *   other training vectors, size [n_rows * n_ws]
 * @param [in] ws_idx indices of traning vectors in the working set, size [n_ws]
 * @param [in] C penalty parameter
 * @param [in] eps tolerance, iterations will stop if the duality gap is smaller
 *  than this value (or if the gap is smaller than 0.1 times the initial gap)
 * @param [out] return_buff, two valies are returned: duality gap and the number
 *   of iterations
 * @param [in] max_iter maximum number of iterations
 */
template <typename math_t, int WSIZE>
__global__ void SmoBlockSolve(math_t *y_array, int n_rows, math_t *alpha,
                              int n_ws, math_t *delta_alpha, math_t *f_array,
                              math_t *kernel, int *ws_idx, math_t C, math_t eps,
                              math_t *return_buff, int max_iter = 10000) {
  typedef Selection::KVPair<math_t, int> Pair;
  typedef cub::BlockReduce<Pair, WSIZE> BlockReduce;
  typedef cub::BlockReduce<math_t, WSIZE> BlockReduceFloat;
  __shared__ union {
    typename BlockReduce::TempStorage pair;
    typename BlockReduceFloat::TempStorage single;
  } temp_storage;

  __shared__ math_t f_u;
  __shared__ int u;
  __shared__ int l;

  __shared__ math_t tmp_u, tmp_l;
  __shared__ math_t Kd[WSIZE];  // diagonal elements of the kernel matrix

  int tid = threadIdx.x;
  int idx = ws_idx[tid];

  // store values in registers
  math_t y = y_array[idx];
  math_t f = f_array[idx];
  math_t a = alpha[idx];
  math_t a_save = a;
  __shared__ math_t diff_end;
  __shared__ math_t diff;

  Kd[tid] = kernel[tid * n_rows + idx];
  int n_iter = 0;

  for (; n_iter < max_iter; n_iter++) {
    // mask values outside of X_upper
    math_t f_tmp = in_upper(a, y, C) ? f : INFINITY;
    Pair pair{f_tmp, tid};
    Pair res = BlockReduce(temp_storage.pair).Reduce(pair, cub::Min(), n_ws);
    if (tid == 0) {
      f_u = res.val;
      u = res.key;
    }
    // select f_max to check stopping condition
    f_tmp = in_lower(a, y, C) ? f : -INFINITY;
    __syncthreads();  // needed because we are reusing the shared memory buffer
    math_t Kui = kernel[u * n_rows + idx];
    math_t f_max =
      BlockReduceFloat(temp_storage.single).Reduce(f_tmp, cub::Max(), n_ws);

    if (tid == 0) {
      // f_max-f_u is used to check stopping condition.
      diff = f_max - f_u;
      if (n_iter == 0) {
        return_buff[0] = diff;
        diff_end = max(eps, 0.1f * diff);
      }
    }
    __syncthreads();
    if (diff < diff_end) {
      break;
    }

    if (f_u < f && in_lower(a, y, C)) {
      f_tmp = (f_u - f) * (f_u - f) / (Kd[tid] + Kd[u] - 2 * Kui);
    } else {
      f_tmp = -INFINITY;
    }
    pair = Pair{f_tmp, tid};
    res = BlockReduce(temp_storage.pair).Reduce(pair, cub::Max(), n_ws);
    if (tid == 0) {
      l = res.key;
    }
    __syncthreads();
    math_t Kli = kernel[l * n_rows + idx];

    // Update alpha
    //
    // We know that 0 <= a <= C
    // We select q so that both delta alpha_u and delta alpha_l stay in this limit.
    if (threadIdx.x == u) tmp_u = y > 0 ? C - a : a;
    if (threadIdx.x == l) {
      tmp_l = y > 0 ? a : C - a;
      tmp_l =
        min(tmp_l, (f - f_u) / (Kd[u] + Kd[l] -
                                2 * Kui));  // note: Kui == Kul for this thread
    }
    __syncthreads();
    math_t q = min(tmp_u, tmp_l);

    if (threadIdx.x == u) a += q * y;
    if (threadIdx.x == l) a -= q * y;
    f += q * (Kui - Kli);
  }
  // save results to global memory before exit
  alpha[idx] = a;
  delta_alpha[tid] = (a - a_save) * y;  // it is actuall y * \Delta \alpha
  // f is recalculated in f_update, therefore we do not need to save that
  return_buff[1] = n_iter;
}
};  // end namespace SVM
};  // end namespace ML
