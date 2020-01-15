/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <string>
#include <type_traits>

#include <cuml/matrix/kernelparams.h>
#include "common/cumlHandle.hpp"
#include "kernelcache.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/gemv.h"
#include "linalg/unary_op.h"
#include "matrix/grammatrix.h"
#include "matrix/kernelfactory.h"
#include "smo_sets.h"
#include "smoblocksolve.h"
#include "utils.h"
#include "workingset.h"
#include "ws_util.h"

#include "common/device_buffer.hpp"
#include "results.h"

namespace ML {
namespace SVM {

/**
 * @brief Solve the quadratic optimization problem using two level decomposition
 * and Sequential Minimal Optimization (SMO).
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
  SmoSolver(const cumlHandle_impl &handle, svmParameter param,
            MLCommon::Matrix::GramMatrixBase<math_t> *kernel)
    : handle(handle),
      n_rows(n_rows),
      C(param.C),
      tol(param.tol),
      kernel(kernel),
      cache_size(param.cache_size),
      nochange_steps(param.nochange_steps),
      epsilon(param.epsilon),
      verbose(param.verbose),
      svmType(param.svmType),
      stream(handle.getStream()),
      return_buff(handle.getDeviceAllocator(), stream, 2),
      alpha(handle.getDeviceAllocator(), stream),
      delta_alpha(handle.getDeviceAllocator(), stream),
      f(handle.getDeviceAllocator(), stream),
      y_label(handle.getDeviceAllocator(), stream) {}

#define SMO_WS_SIZE 1024
  /**
   * @brief Solve the quadratic optimization problem.
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
    // Prepare data structures for SMO
    WorkingSet<math_t> ws(handle, stream, n_rows, SMO_WS_SIZE, svmType,
                          verbose);
    n_ws = ws.GetSize();
    Initialize(&y, n_rows, n_cols);
    KernelCache<math_t> cache(handle, x, n_rows, n_cols, n_ws, kernel,
                              cache_size, svmType, verbose);
    // Init counters
    max_outer_iter = GetDefaultMaxIter(n_train, max_outer_iter);
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
        y, n_train, alpha.data(), n_ws, delta_alpha.data(), f.data(), cacheTile,
        cache.GetWsIndices(), C, tol, return_buff.data(), max_inner_iter,
        svmType, cache.GetColIdxMap());

      CUDA_CHECK(cudaPeekAtLastError());

      MLCommon::updateHost(host_return_buff, return_buff.data(), 2, stream);

      UpdateF(f.data(), n_rows, delta_alpha.data(), cache.GetUniqueSize(),
              cacheTile);

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
    Results<math_t> res(handle, x, y, n_rows, n_cols, C, svmType);
    res.Get(alpha.data(), f.data(), dual_coefs, n_support, idx, x_support, b);
    ReleaseBuffers();
  }

  /**
   * @brief Update the f vector after a block solve step.
   *
   * \f[ f_i = f_i + \sum_{k\in WS} K_{i,k} * \Delta \alpha_k, \f]
   * where i = [0..n_train-1], WS is the set of workspace indices,
   * and \f$K_{i,k}\f$ is the kernel function evaluated for training vector x_i and workspace vector x_k.
   *
   * @param f size [n_train]
   * @param n_rows
   * @param delta_alpha size [n_ws]
   * @param n_ws
   * @param cacheTile kernel function evaluated for the following set K[X,x_ws],
   *   size [n_rows, n_ws]
   * @param cublas_handle
   */
  void UpdateF(math_t *f, int n_rows, const math_t *delta_alpha, int n_ws,
               const math_t *cacheTile) {
    // multipliers used in the equation : f = 1*cachtile * delta_alpha + 1*f
    math_t one = 1;
    CUBLAS_CHECK(MLCommon::LinAlg::cublasgemv(
      handle.getCublasHandle(), CUBLAS_OP_N, n_rows, n_ws, &one, cacheTile,
      n_rows, delta_alpha, 1, &one, f, 1, stream));
    if (svmType == EPSILON_SVR) {
      // SVR has doubled the number of trainig vectors and we need to update
      // alpha for both batches individually
      CUBLAS_CHECK(MLCommon::LinAlg::cublasgemv(
        handle.getCublasHandle(), CUBLAS_OP_N, n_rows, n_ws, &one, cacheTile,
        n_rows, delta_alpha, 1, &one, f + n_rows, 1, stream));
    }
  }

  /** @brief Initialize the problem to solve.
   *
   * Both SVC and SVR are solved as a classification problem.
   * The optimization target (W) does not appear directly in the SMO
   * formulation, only its derivative through f (optimality indicator vector):
   * \f[ f_i = y_i \frac{\partial W }{\partial \alpha_i}. \f]
   *
   * The f_i values are initialized here, and updated at every solver iteration
   * when alpha changes. The update step is the same for SVC and SVR, only the
   * init step differs.
   *
   * Additionally, we zero init the dual coefficients (alpha), and initialize
   * class labels for SVR.
   *
   * @parameter [inout] y on entry class labels or target values,
   *    on exit device pointer to class labels
   * @parameter [in] n_rows
   * @parameter [in] n_cols
   */
  void Initialize(math_t **y, int n_rows, int n_cols) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    n_train = (svmType == EPSILON_SVR) ? n_rows * 2 : n_rows;
    ResizeBuffers(n_train, n_cols);
    // Zero init alpha
    CUDA_CHECK(
      cudaMemsetAsync(alpha.data(), 0, n_train * sizeof(math_t), stream));
    // Init f (and also class labels for SVR)
    switch (svmType) {
      case C_SVC:
        SvcInit(*y);
        break;
      case EPSILON_SVR:
        SvrInit(*y, n_rows, y_label.data(), f.data());
        // We return the pointer to the class labels (the target values are
        // not needed anymore, they are incorporated in f).
        *y = y_label.data();
        break;
      default:
        THROW("SMO initialization not implemented SvmType=%d", svmType);
    }
  }

  /** @brief Initialize Support Vector Classification
   *
   * We would like to maximize the following quantity
   * \f[ W(\mathbf{\alpha}) = -\mathbf{\alpha}^T \mathbf{1}
   *   + \frac{1}{2} \mathbf{\alpha}^T Q \mathbf{\alpha}, \f]
   *
   * We initialize f as:
   * \f[ f_i = y_i \frac{\partial W(\mathbf{\alpha})}{\partial \alpha_i} =
   *          -y_i +   y_j \alpha_j K(\mathbf{x}_i, \mathbf{x}_j) \f]
   *
   * @param [in] y device pointer of class labels size [n_rows]
   */
  void SvcInit(const math_t *y) {
    MLCommon::LinAlg::unaryOp(
      f.data(), y, n_rows, [] __device__(math_t y) { return -y; }, stream);
  }

  /**
   * @brief Initializes the solver for epsilon-SVR.
   *
   * For regression we are optimizing the following quantity
   * \f[
   * W(\alpha^+, \alpha^-) =
   * \epsilon \sum_{i=1}^l (\alpha_i^+ + \alpha_i^-)
   * - \sum_{i=1}^l yc_i (\alpha_i^+ - \alpha_i^-)
   * + \frac{1}{2} \sum_{i,j=1}^l
   *   (\alpha_i^+ - \alpha_i^-)(\alpha_j^+ - \alpha_j^-) K(\bm{x}_i, \bm{x}_j)
   * \f]
   *
   * Then \f$ f_i = y_i \frac{\partial W(\alpha}{\partial \alpha_i} \f$
   *      \f$     = yc_i*epsilon - yr_i \f$
   *
   * Additionally we set class labels for the training vectors.
   *
   * References:
   * [1] B. Schölkopf et. al (1998): New support vector algorithms,
   *     NeuroCOLT2 Technical Report Series, NC2-TR-1998-031, Section 6
   * [2] A.J. Smola, B. Schölkopf (2004): A tutorial on support vector
   *     regression, Statistics and Computing 14, 199–222
   * [3] Orchel M. (2011) Support Vector Regression as a Classification Problem
   *     with a Priori Knowledge in the Form of Detractors,
   *     Man-Machine Interactions 2. Advances in Intelligent and Soft Computing,
   *     vol 103
   *
   * @param [in] yr device pointer with values for regression, size [n_rows]
   * @param [in] n_rows
   * @param [out] yc device pointer to classes associated to the dual
   *     coefficients, size [n_rows*2]
   * @param [out] f device pointer f size [n_rows*2]
   */
  void SvrInit(const math_t *yr, int n_rows, math_t *yc, math_t *f) {
    // Init class labels to [1, 1, 1, ..., -1, -1, -1, ...]
    thrust::device_ptr<math_t> yc_ptr(yc);
    thrust::constant_iterator<math_t> one(1);
    thrust::copy(thrust::cuda::par.on(stream), one, one + n_rows, yc_ptr);
    thrust::constant_iterator<math_t> minus_one(-1);
    thrust::copy(thrust::cuda::par.on(stream), minus_one, minus_one + n_rows,
                 yc_ptr + n_rows);

    // f_i = epsilon - y_i, for i \in [0..n_rows-1]
    math_t epsilon = this->epsilon;
    MLCommon::LinAlg::unaryOp(
      f, yr, n_rows, [epsilon] __device__(math_t y) { return epsilon - y; },
      stream);

    // f_i = epsilon - y_i, for i \in [n_rows..2*n_rows-1]
    MLCommon::LinAlg::unaryOp(
      f + n_rows, yr, n_rows,
      [epsilon] __device__(math_t y) { return -epsilon - y; }, stream);
  }

 private:
  const cumlHandle_impl &handle;
  cudaStream_t stream;

  int n_rows = 0;  //!< training data number of rows
  int n_cols = 0;  //!< training data number of columns
  int n_ws = 0;    //!< size of the working set
  int n_train =
    0;  //!< number of training vectors (including duplicates for SVR)

  // Buffers for the domain [n_train]
  MLCommon::device_buffer<math_t> alpha;    //!< dual coordinates
  MLCommon::device_buffer<math_t> f;        //!< optimality indicator vector
  MLCommon::device_buffer<math_t> y_label;  //!< extra label for regression

  // Buffers for the working set [n_ws]
  //! change in alpha parameter during a blocksolve step
  MLCommon::device_buffer<math_t> delta_alpha;

  // Buffers to return some parameters from the kernel (iteration number, and
  // convergence information)
  MLCommon::device_buffer<math_t> return_buff;
  math_t host_return_buff[2];

  math_t C;
  math_t tol;      //!< tolerance for stopping condition
  math_t epsilon;  //!< epsilon parameter for epsiolon-SVR

  MLCommon::Matrix::GramMatrixBase<math_t> *kernel;
  float cache_size;  //!< size of kernel cache in MiB

  SvmType svmType;  ///!< Type of the SVM problem to solve

  // Variables to track convergence of training
  math_t diff_prev;
  int n_small_diff;
  int nochange_steps;

  bool CheckStoppingCondition(math_t diff) {
    // TODO improve stopping condition to detect oscillations, see Issue #947
    bool keep_going = true;
    if (abs(diff - diff_prev) < 0.001 * tol) {
      n_small_diff++;
    } else {
      diff_prev = diff;
      n_small_diff = 0;
    }
    if (n_small_diff > nochange_steps) {
      if (verbose) {
        std::cout << "SMO error: Stopping due to unchanged diff over "
                  << nochange_steps << " consecutive steps\n";
      }
      keep_going = false;
    }
    if (diff < tol) keep_going = false;
    if (isnan(diff)) {
      std::string txt = "SMO error: NaN found during fitting.";
      if (std::is_same<float, math_t>::value) {
        txt +=
          " This might be caused by floating point overflow. In such case using"
          " fp64 could help. Alternatively, try gamma='scale' kernel"
          " parameter.";
      }
      THROW(txt.c_str());
    }
    return keep_going;
  }

  /// Return the number of maximum iterations.
  int GetDefaultMaxIter(int n_train, int max_outer_iter) {
    if (max_outer_iter == -1) {
      max_outer_iter = n_train < std::numeric_limits<int>::max() / 100
                         ? n_train * 100
                         : std::numeric_limits<int>::max();
      max_outer_iter = max(100000, max_outer_iter);
    }
    // else we have user defined iteration count which we do not change
    return max_outer_iter;
  }

  void ResizeBuffers(int n_train, int n_cols) {
    // This needs to know n_rows, therefore it can be only called during solve
    alpha.resize(n_train, stream);
    f.resize(n_train, stream);
    delta_alpha.resize(n_ws, stream);
    if (svmType == EPSILON_SVR) y_label.resize(n_train, stream);
  }

  void ReleaseBuffers() {
    alpha.release(stream);
    delta_alpha.release(stream);
    f.release(stream);
    y_label.release(stream);
  }
};

};  // end namespace SVM
};  // end namespace ML
