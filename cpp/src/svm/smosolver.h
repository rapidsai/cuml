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
#include <math.h>
#include <iostream>
#include <limits>

#include "common/cumlHandle.hpp"
#include "gram/grammatrix.h"
#include "gram/kernelfactory.h"
#include "gram/kernelparams.h"
#include "kernelcache.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/gemv.h"
#include "linalg/unary_op.h"
#include "smo_sets.h"
#include "smoblocksolve.h"
#include "workingset.h"
#include "ws_util.h"

#include "common/device_buffer.hpp"
#include "results.h"

namespace ML {
namespace SVM {

/**
 * Solve the quadratic optimization problem using two level decomposition and
 * Sequential Minimal Optimization (SMO).
 *
 * The general decomposition idea by Osuna is to choose q examples from all the
 * training examples, and solve the QP problem for this subset (discussed in
 * section 11.2 by Joachims [1]). SMO is the extreme case where we choose q=2.
 *
 * Here we follow [2] and [3] and use two level decomposition. First we set
 * q_1=1024, and solve the QP sub-problem for that (let's call it QP1). This is
 * the outer iteration, implemented in SmoSolver::Solve.
 *
 * To solve QP1, we use another decomposition, specifically the SMO (q_2 = 2),
 * which is implemented in SmoBlockSolve.
 *
 * References:
 * - [1] Joachims, T. Making large-scale support vector machine learning
 *      practical. In B. Scholkopf, C. Burges, & A. Smola (Eds.), Advances in
 *      kernel methods: Support vector machines. Cambridge, MA: MIT Press (1998)
 * - [2] J. Vanek et al. A GPU-Architecture Optimized Hierarchical Decomposition
 *      Algorithm for Support VectorMachine Training, IEEE Transactions on
 *      Parallel and Distributed Systems, vol 28, no 12, 3330, (2017)
 * - [3] Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs, Journal
 *      of Machine Learning Research, 19, 1-5 (2018)
 */
template <typename math_t>
class SmoSolver {
 public:
  bool verbose = false;
  SmoSolver(const cumlHandle_impl &handle, math_t C, math_t tol,
            MLCommon::GramMatrix::GramMatrixBase<math_t> *kernel,
            float cache_size = 200)
    : handle(handle),
      n_rows(n_rows),
      C(C),
      tol(tol),
      kernel(kernel),
      cache_size(cache_size),
      stream(handle.getStream()),
      return_buff(handle.getDeviceAllocator(), stream, 2),
      alpha(handle.getDeviceAllocator(), stream),
      delta_alpha(handle.getDeviceAllocator(), stream),
      f(handle.getDeviceAllocator(), stream) {}

#define SMO_WS_SIZE 1024
  /**
   * Solve the quadratic optimization problem.
   *
   * The output arrays (dual_coefs, x_support, idx) will be allocated on the
   * device, they should be unallocated on entry.
   *
   * @param [in] x training vectors in column major format, size [n_rows x n_cols]
   * @param [in] n_rows number of rows (training vectors)
   * @param [in] n_cols number of columns (features)
   * @param [in] y labels (values +/-1), size [n_rows]
   * @param [out] dual_coefs, size [n_support] on exit
   * @param [out] n_support number of support vectors
   * @param [out] x_support support vectors in column major format, size [n_support, n_cols]
   * @param [out] idx the original training set indices of the support vectors, size [n_support]
   * @param [out] b scalar constant for the decision function
   * @param [in] max_out_iter maximum number of outer iteration (default 100 * n_rows)
   * @param [in] xm_inner_iter maximum number of inner iterations (default 10000)
   */
  void Solve(math_t *x, int n_rows, int n_cols, math_t *y, math_t **dual_coefs,
             int *n_support, math_t **x_support, int **idx, math_t *b,
             int max_outer_iter = -1, int max_inner_iter = 10000) {
    if (max_outer_iter == -1) {
      max_outer_iter = n_rows < std::numeric_limits<int>::max() / 100
                         ? n_rows * 100
                         : std::numeric_limits<int>::max();
      max_outer_iter = max(100000, max_outer_iter);
    }

    WorkingSet<math_t> ws(handle, stream, n_rows, SMO_WS_SIZE);
    int n_ws = ws.GetSize();
    ResizeBuffers(n_rows, n_cols, n_ws);
    Initialize(y);

    KernelCache<math_t> cache(handle, x, n_rows, n_cols, n_ws, kernel,
                              cache_size);

    int n_iter = 0;
    int n_inner_iter = 0;
    diff_prev = 0;
    n_small_diff = 0;
    bool keep_going = true;

    while (n_iter < max_outer_iter && keep_going) {
      CUDA_CHECK(
        cudaMemsetAsync(delta_alpha.data(), 0, n_ws * sizeof(math_t), stream));
      ws.Select(f.data(), alpha.data(), y, C);

      math_t *cacheTile = cache.GetTile(ws.GetIndices());

      SmoBlockSolve<math_t, SMO_WS_SIZE><<<1, n_ws, 0, stream>>>(
        y, n_rows, alpha.data(), n_ws, delta_alpha.data(), f.data(), cacheTile,
        ws.GetIndices(), C, tol, return_buff.data(), max_inner_iter);

      CUDA_CHECK(cudaPeekAtLastError());

      updateHost(host_return_buff, return_buff.data(), 2, stream);

      UpdateF(f.data(), n_rows, delta_alpha.data(), n_ws, cacheTile);

      CUDA_CHECK(cudaStreamSynchronize(stream));

      math_t diff = host_return_buff[0];
      keep_going = CheckStoppingCondition(diff);

      n_inner_iter += host_return_buff[1];
      n_iter++;
      if (verbose && n_iter % 500 == 0) {
        std::cout << "SMO iteration " << n_iter << ", diff " << diff << "\n";
      }
    }

    if (verbose) {
      std::cout << "SMO solver finished after " << n_iter
                << " outer iterations, " << n_inner_iter
                << " total inner iterations, and diff " << diff_prev << "\n";
    }
    Results<math_t> res(handle, x, y, n_rows, n_cols, C);
    res.Get(alpha.data(), f.data(), dual_coefs, n_support, idx, x_support, b);
    ReleaseBuffers();
  }

  /**
   * Update the f vector after a block solve step.
   *
   * \f[ f_i = f_i + \sum_{k\in WS} K_{i,k} * \Delta \alpha_k, \f]
   * where i = [0..n_rows-1], WS is the set of workspace indices,
   * and \f$K_{i,k}\f$ is the kernel function evaluated for training vector x_i and workspace vector x_k.
   *
   * @param f size [n_rows]
   * @param n_rows
   * @param delta_alpha size [n_ws]
   * @param n_ws
   * @param cacheTile kernel function evaluated for the following set K[X,x_ws], size [n_rows, n_ws]
   * @param cublas_handle
   */
  void UpdateF(math_t *f, int n_rows, const math_t *delta_alpha, int n_ws,
               const math_t *cacheTile) {
    math_t one =
      1;  // multipliers used in the equation : f = 1*cachtile * delta_alpha + 1*f
    CUBLAS_CHECK(LinAlg::cublasgemv(handle.getCublasHandle(), CUBLAS_OP_N,
                                    n_rows, n_ws, &one, cacheTile, n_rows,
                                    delta_alpha, 1, &one, f, 1, stream));
  }

  /// Initialize the values of alpha and f
  void Initialize(math_t *y) {
    // we initialize alpha_i = 0 and
    // f_i = -y_i
    CUDA_CHECK(
      cudaMemsetAsync(alpha.data(), 0, n_rows * sizeof(math_t), stream));
    LinAlg::unaryOp(
      f.data(), y, n_rows, [] __device__(math_t y) { return -y; }, stream);
  }

 private:
  const cumlHandle_impl &handle;
  cudaStream_t stream;

  int n_rows = 0;  //!< training data number of rows
  int n_cols = 0;  //!< training data number of columns
  int n_ws = 0;    //!< size of the working set

  // Buffers for the domain [n_rows]
  MLCommon::device_buffer<math_t> alpha;  //!< dual coordinates
  MLCommon::device_buffer<math_t> f;      //!< optimality indicator vector

  // Buffers for the working set [n_ws]
  //! change in alpha parameter during a blocksolve step
  MLCommon::device_buffer<math_t> delta_alpha;

  // Buffers to return some parameters from the kernel (iteration number, and
  // convergence information)
  MLCommon::device_buffer<math_t> return_buff;
  math_t host_return_buff[2];

  math_t C;
  math_t tol;  //!< tolerance for stopping condition

  MLCommon::GramMatrix::GramMatrixBase<math_t> *kernel;
  float cache_size;  //!< size of kernel cache in MiB

  // Variables to track convergence of training
  math_t diff_prev;
  int n_small_diff;

  bool CheckStoppingCondition(math_t diff) {
    // TODO improve stopping condition to detect oscillations
    bool keep_going = true;
    if (abs(diff - diff_prev) < 0.001 * tol) {
      n_small_diff++;
    } else {
      diff_prev = diff;
      n_small_diff = 0;
    }
    if (diff < tol || n_small_diff > 10) {
      keep_going = false;
    }
    // ASSERT(!isnan(diff), "SMO: NaN found during fitting")
    if (isnan(diff)) {
      std::cout << "SMO error: NaN found during fitting\n";
      keep_going = false;
    }
    return keep_going;
  }

  void ResizeBuffers(int n_rows, int n_cols, int n_ws) {
    // This needs to know n_ws, therefore it can be only called during the solve step
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    this->n_ws = n_ws;
    alpha.resize(n_rows, stream);
    f.resize(n_rows, stream);
    delta_alpha.resize(n_ws, stream);
  }

  void ReleaseBuffers() {
    alpha.release(stream);
    delta_alpha.release(stream);
    f.release(stream);
  }
};

};  // end namespace SVM
};  // end namespace ML
