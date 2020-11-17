/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <iostream>
#include <limits>
#include <numeric>

#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/gemv.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <cache/cache_util.cuh>
#include <common/allocatorAdapter.hpp>
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>
#include <common/host_buffer.hpp>
#include <cub/cub.cuh>
#include <cuml/common/logger.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/cholesky_r1_update.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/unary_op.cuh>

namespace ML {
namespace Solver {
namespace Lars {

/**
 * @brief Select the largest element from the inactive working set.
 *
 * The inactive set consist of cor[n_active..n-1]. This function returns the
 * index of the most correlated element. The value of the largest element is
 * returned in cj.
 *
 * @param n_active number of active elements (n_active <= n )
 * @param n number of elements in vector cor
 * @param correlation device array of correlations, size [n]
 * @param cj host pointer to return the value of the largest element
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
idx_t selectMostCorrelated(idx_t n_active, idx_t n, math_t* correlation,
                           math_t* cj,
                           MLCommon::device_buffer<math_t>& workspace,
                           cudaStream_t stream) {
  const idx_t align_bytes = 16 * sizeof(math_t);
  // We might need to start a few elements earlier to ensure that the unary
  // op has aligned access for vectorized load.
  int start = raft::alignDown<idx_t>(n_active, align_bytes) / sizeof(math_t);
  raft::linalg::unaryOp(
    workspace.data(), correlation + start, n,
    [] __device__(math_t a) { return abs(a); }, stream);
  thrust::device_ptr<math_t> ptr(workspace.data() + n_active - start);
  auto max_ptr =
    thrust::max_element(thrust::cuda::par.on(stream), ptr, ptr + n - n_active);
  raft::update_host(cj, max_ptr.get(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return n_active + (max_ptr - ptr);  // return the index
}

/**
 * @brief Move feature at idx=j into the active set.
 *
 * We have an active set with n_active elements, and an inactive set with
 * n_cols - n_active elements. The matrix X [n_samples, n_features] is
 * partitioned in a way that the first n_active columns store the active set.
 * Similarily the vectors correlation and indices are partitioned in a way
 * that the first n_active elements belong to the active set:
 * - active set:  X[:,:n_active], correlation[:n_active], indices[:n_active]
 * - inactive set: X[:,n_active:], correlation[n_active:], indices[n_active:].
 *
 * This function moves the feature column X[:,idx] into the active set by
 * replacing the first inactive element with idx. The indices and correlation
 * vectors are modified accordinly. The sign array is updated with the sign
 * of correlation[n_active].
 *
 * @param handle cuBLAS handle
 * @param n_active number of active elements, will be increased by one after
 *     we move the new element j into the active set
 * @param j index of the new element (n_active <= j < n_cols)
 * @param X device array of feature vectors in column major format, size
 *     [n_cols * ld_X]
 * @param n_rows number of training vectors
 * @param n_cols number of features
 * @param ld_X leading dimension of X
 * @param correlations device array of correlations, size [n_cols]
 * @param indices host array of indices, size [n_cols]
 * @param G device pointer of Gram matrix (or nullptr), size [n_cols * ld_G]
 * @param ld_G leading dimension of G
 * @param sign device pointer to sign array, size[n]
 */
template <typename math_t, typename idx_t = int>
void moveToActive(cublasHandle_t handle, idx_t* n_active, idx_t j, math_t* X,
                  idx_t n_rows, idx_t n_cols, idx_t ld_X, math_t* cor,
                  idx_t* indices, math_t* G, idx_t ld_G, math_t* sign,
                  cudaStream_t stream) {
  idx_t idx_free = *n_active;
  std::swap(indices[idx_free], indices[j]);
  if (G) {
    CUBLAS_CHECK(raft::linalg::cublasSwap(handle, n_cols, G + ld_G * idx_free,
                                          1, G + ld_G * j, 1, stream));
    CUBLAS_CHECK(raft::linalg::cublasSwap(handle, n_cols, G + idx_free, ld_G,
                                          G + j, ld_G, stream));
  } else {
    // Only swap X if G is nullptr. Only in that case will we use the feature
    // columns, otherwise all the necessary information is already there in G.
    CUBLAS_CHECK(raft::linalg::cublasSwap(handle, n_rows, X + ld_X * idx_free,
                                          1, X + ld_X * j, 1, stream));
  }
  // sign[n_active] = sign(c[j])
  raft::linalg::unaryOp(
    sign + idx_free, cor + j, 1,
    [] __device__(math_t c) -> math_t {
      // return the sign of c
      return (math_t(0) < c) - (c < math_t(0));
    },
    stream);
  // swap (c[idx_free], c[j])
  CUBLAS_CHECK(
    raft::linalg::cublasSwap(handle, 1, cor + idx_free, 1, cor + j, 1, stream));
  (*n_active)++;
}

/**
 * @brief Update the Cholesky decomposition of the Gram matrix of the active set
 *
 * G0 = X.T * X, Gram matrix without signs. We use the part that corresponds to
 * the active set, [n_A x n_A]
 *
 * At each step on the LARS path we add one column to the active set, therefore
 * the Gram matrix grows incrementally. We update the Cholesky decomposition
 * G0 = U.T * U.
 *
 * The Cholesky decomposition can use the same storage as G0, if the input
 * pointers are same.
 *
 * @param handle RAFT handle
 * @param n_active number of active elements
 * @param X device array  of feature vectors in column major format, size
 *     [n_rows * n_cols]
 * @praam n_rows number of training vectors
 * @param n_cols number of features
 * @param ld_X leading dimension of X (stride of columns)
 * @param U device pointer to the Cholesky decomposition of G0,
 *     size [n_cols * n_cols]
 * @param ld_U leading dimension of U
 * @param G0 device pointer to Gram matrix G0 = X.T*X (can be nullptr),
 *     size [n_cols * n_cols].
 * @param workspace workspace for the Cholesky update
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void updateCholesky(const raft::handle_t& handle, idx_t n_active,
                    const math_t* X, idx_t n_rows, idx_t n_cols, idx_t ld_X,
                    math_t* U, idx_t ld_U, const math_t* G0, idx_t ld_G,
                    MLCommon::device_buffer<math_t>& workspace,
                    cudaStream_t stream) {
  const cublasFillMode_t fillmode = CUBLAS_FILL_MODE_UPPER;
  if (G0 == nullptr) {
    // Calculate the new column of G0. It is stored in U.
    math_t* G_row = U + (n_active - 1) * ld_U;
    const math_t* X_row = X + (n_active - 1) * ld_X;
    math_t one = 1;
    math_t zero = 0;
    CUBLAS_CHECK(raft::linalg::cublasgemv(
      handle.get_cublas_handle(), CUBLAS_OP_T, n_rows, n_cols, &one, X, n_rows,
      X_row, 1, &zero, G_row, 1, stream));
  } else if (G0 != U) {
    // Copy the new column of G0 into U, because the factorization works in
    // place.
    raft::copy(U + (n_active - 1) * ld_U, G0 + (n_active - 1) * ld_G, n_active,
               stream);
  }  // Otherwise the new data is already in place in U.

  // Update the Cholesky decomposition
  int n_work = workspace.size();
  if (n_work == 0) {
    // Query workspace size and allocate it
    raft::linalg::choleskyRank1Update(handle, U, n_active, ld_U, nullptr,
                                      &n_work, fillmode, stream);
    workspace.resize(n_work, stream);
  }
  raft::linalg::choleskyRank1Update(handle, U, n_active, ld_U, workspace.data(),
                                    &n_work, fillmode, stream);
}

/**
 * @brief Solve for ws = S * GA^(-1) * 1_A  using a Cholesky decomposition.
 *
 * See calcEquiangularVec for more details on the formulas. In this function we
 * calculate ws = S * (S * G0 * S)^{-1} 1_A = G0^{-1} (S 1_A) = G0^{-1} sign_A.
 *
 * @param handle RAFT handle
 * @param n_active number of active elements
 * @param n_cols number of features
 * @parma sign array with sign of the active set, size [n_cols]
 * @param U device pointer to the Cholesky decomposition of G0,
 *     size [n_cols * n_cols]
 * @param ws device pointer, size [n_active]
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void calcW0(const raft::handle_t& handle, idx_t n_active, idx_t n_cols,
            const math_t* sign, const math_t* U, idx_t ld_U, math_t* ws,
            cudaStream_t stream) {
  const cublasFillMode_t fillmode = CUBLAS_FILL_MODE_UPPER;

  // First we alculate x by solving equation U.T x = sign_A.
  raft::copy(ws, sign, n_active, stream);
  math_t alpha = 1;
  CUBLAS_CHECK(raft::linalg::cublastrsm(
    handle.get_cublas_handle(), CUBLAS_SIDE_LEFT, fillmode, CUBLAS_OP_T,
    CUBLAS_DIAG_NON_UNIT, n_active, 1, &alpha, U, ld_U, ws, ld_U, stream));

  // w0 stores x, the solution of U.T x = sign_A. Now we solve U * ws = x
  CUBLAS_CHECK(raft::linalg::cublastrsm(
    handle.get_cublas_handle(), CUBLAS_SIDE_LEFT, fillmode, CUBLAS_OP_N,
    CUBLAS_DIAG_NON_UNIT, n_active, 1, &alpha, U, ld_U, ws, ld_U, stream));
  // Now ws = G0^(-1) sign_A = S GA^{-1} 1_A.
}

/**
 * @brief Calculate A = (1_A * GA^{-1} * 1_A)^{-1/2}.
 *
 * See calcEquiangularVec for more details on the formulas.
 *
 * @param handle RAFT handle
 * @param A device pointer to store the result
 * @param n_active number of active elements
 * @param sign array with sign of the active set, size [n_cols]
 * @param ws device pointer, size [n_active]
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void calcA(const raft::handle_t& handle, math_t* A, idx_t n_active,
           const math_t* sign, const math_t* ws, cudaStream_t stream) {
  // Calculate sum (w) = sum(ws * sign)
  auto multiply = [] __device__(math_t w, math_t s) { return w * s; };
  raft::linalg::mapThenSumReduce(A, n_active, multiply, stream, ws, sign);
  // Calc Aa = 1 / sqrt(sum(w))
  raft::linalg::unaryOp(
    A, A, 1, [] __device__(math_t a) { return 1 / sqrt(a); }, stream);
}

/**
 * @brief Calculate the equiangular vector u, w and A according to [1].
 *
 * We introduce the following variables (Python like indexing):
 * - n_A number of elements in the active set
 * - S = diag(sign_A): diagonal matrix with the signs, size [n_A x n_A]
 * - X_A = X[:,:n_A] * S, column vectors of the active set size [n_A x n_A]
 * - G0 = X.T * X, Gram matrix without signs. We just use the part that
 *   corresponds to the active set, [n_A x n_A]
 * - GA = X_A.T * X_A is the Gram matrix of the active set, size [n_A x n_A]
 *   GA = S * G0[:n_A, :n_A] * S
 * - 1_A = np.ones(n_A)
 * - A = (1_A * GA^{-1} * 1_A)^{-1/2}, scalar, see eq (2.5) in [1]
 * - w = A GA^{-1} * 1_A, vector of size [n_A] see eq (2.6) in [1]
 * - ws = S * w, vector of size [n_A]
 *
 * The equiangular vector can be expressed the following way (equation 2.6):
 * u = X_A * w = X[:,:n_A] S * w = X[:,:n_A] * ws.
 *
 * The equiangular vector later appears only in an expression like X.T u, which
 * can be reformulated as X.T u = X.T X[:,:n_A] S * w = G[:n_A,:n_A] * ws.
 * If the gram matrix is given, then we do not need to calculate u, it will be
 * sufficient to calculate ws and A.
 *
 * We use Cholesky decomposition G0 = U.T * U to solve to calculate A and w
 * which depend on GA^{-1}.
 *
 * References:
 *  [1] B. Efron, T. Hastie, I. Johnstone, R Tibshirani, Least Angle Regression
 *  The Annals of Statistics (2004) Vol 32, No 2, 407-499
 *  http://statweb.stanford.edu/~tibs/ftp/lars.pdf
 *
 * @param handle RAFT handle
 * @param n_active number of active elements
 * @param X device array  of feature vectors in column major format, size
 *     [n_rows * n_cols]
 * @praam n_rows number of training vectors
 * @param n_cols number of features
 * @parma sign array with sign of the active set, size [n_cols]
 * @param U device pointer to the Cholesky decomposition of G0,
 *     size [n_cols * n_cols]
 * @param G0 device pointer to Gram matrix G0 = X.T*X (can be nullptr),
 *     size [n_cols * n_cols]. Note the difference between G0 and
 *     GA = X_A.T * X_A
 * @param workspace workspace for the Cholesky update
 * @param w device pointer, size [n_active]
 * @param A device pointer to a scalar
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void calcEquiangularVec(const raft::handle_t& handle, idx_t n_active, math_t* X,
                        idx_t n_rows, idx_t n_cols, idx_t ld_X, math_t* sign,
                        math_t* U, idx_t ld_U, math_t* G0, idx_t ld_G,
                        MLCommon::device_buffer<math_t>& workspace, math_t* ws,
                        math_t* A, math_t* u_eq, cudaStream_t stream) {
  // Since we added a new vector to the active set, we update the Cholesky
  // decomposition (U)
  updateCholesky(handle, n_active, X, n_rows, n_cols, ld_X, U, ld_U, G0, ld_G,
                 workspace, stream);

  // Calculate ws = S GA^{-1} 1_A using U
  calcW0(handle, n_active, n_cols, sign, U, ld_U, ws, stream);

  calcA(handle, A, n_active, sign, ws, stream);

  // ws *= Aa
  raft::linalg::unaryOp(
    ws, ws, n_active, [A] __device__(math_t w) { return (*A) * w; }, stream);

  if (G0 == nullptr) {
    // Calculate u_eq only in the case if the Gram matrix is not stored.
    math_t one = 1;
    math_t zero = 0;
    CUBLAS_CHECK(raft::linalg::cublasgemv(
      handle.get_cublas_handle(), CUBLAS_OP_N, n_rows, n_active, &one, X, ld_X,
      ws, 1, &zero, u_eq, 1, stream));
  }
}

/**
 * @brief Calculate the maximum step size (gamma) in the equiangular direction.
 *
 * Let mu = X beta.T be the current prediction vector. The modified solution
 * after taking step gamma is defined as mu' = mu + gamma mu. With this
 * solution the correlation of the covariates in the active set will decrease
 * equally, to a new value |c_j(gamma)| = Cmax - gamma A. At the same time
 * the correlation of the values in the inactive set changes according to the
 * following formula: c_j(gamma) = c_j - gamma a_j. We increase gamma until
 * one of correlations from the inactive set becomes equal with the
 * correlation from the active set.
 *
  * References:
 *  [1] B. Efron, T. Hastie, I. Johnstone, R Tibshirani, Least Angle Regression
 *  The Annals of Statistics (2004) Vol 32, No 2, 407-499
 *  http://statweb.stanford.edu/~tibs/ftp/lars.pdf
 *
 * @param handle RAFT handle
 * @param max_iter maximum number of iterations
 * @param n_active size of the active set (n_active <= max_iter <= n_cols)
 * @param cj value of the maximum correlation
 * @param A device pointer to a scalar, as defined by eq 2.5 in [1]
 * @param cor device pointer to correlation vector, size [n_active]
 * @param G device pointer to Gram matrix of the active set (without signs)
 *    size [n_active * ld_G]
 * @param X device array of training vectors in column major format,
 *     size [n_rows * n_cols]. Only used if the gram matrix is not avaiable.
 * @param u device pointer to equiangular vector size [n_rows]. Only used if the
 *     Gram matrix G is not available.
 * @param ws device pointer to the ws vector defined in calcEquiangularVec,
 *    size [n_active]
 * @param stream CUDA stream
 */
template <typename math_t, typename idx_t = int>
void calcMaxStep(const raft::handle_t& handle, idx_t max_iter, idx_t n_rows,
                 idx_t n_cols, idx_t n_active, math_t cj, const math_t* A,
                 math_t* cor, const math_t* G, idx_t ld_G, const math_t* X,
                 idx_t ld_X, const math_t* u, const math_t* ws, math_t* gamma,
                 math_t* a_vec, cudaStream_t stream) {
  // In the active set each element has the same correlation, whose absolute
  // value is given by Cmax.
  math_t Cmax = std::abs(cj);
  if (n_active == max_iter) {
    // Last iteration, the inactive set is empty we use equation (2.21)
    raft::linalg::unaryOp(
      gamma, A, 1, [Cmax] __device__(math_t A) { return Cmax / A; }, stream);
  } else {
    const int n_inactive = max_iter - n_active;
    if (G == nullptr) {
      // Calculate a = X.T[:,n_active:] * u                              (2.11)
      math_t one = 1;
      math_t zero = 0;
      CUBLAS_CHECK(raft::linalg::cublasgemv(
        handle.get_cublas_handle(), CUBLAS_OP_T, n_rows, n_inactive, &one,
        X + n_active * ld_X, ld_X, u, 1, &zero, a_vec, 1, stream));
    } else {
      // Calculate a = X.T[:,n_A:] * u = X.T[:, n_A:] * X[:,:n_A] * ws
      //             = G[n_A:,:n_A] * ws                                 (2.11)
      math_t one = 1;
      math_t zero = 0;
      CUBLAS_CHECK(raft::linalg::cublasgemv(
        handle.get_cublas_handle(), CUBLAS_OP_N, n_inactive, n_active, &one,
        G + n_active, ld_G, ws, 1, &zero, a_vec, 1, stream));
    }
    const math_t tiny = std::numeric_limits<math_t>::min();
    const math_t huge = std::numeric_limits<math_t>::max();
    //
    // gamma = min^+_{j \in inactive} {(Cmax - cor_j) / (A-a_j),
    //                                 (Cmax + cor_j) / (A+a_j)}         (2.13)
    auto map = [Cmax, A, tiny, huge] __device__(math_t c, math_t a) -> math_t {
      math_t tmp1 = (Cmax - c) / (*A - a + tiny);
      math_t tmp2 = (Cmax + c) / (*A + a + tiny);
      // We consider only positive elements while we search for the minimum
      math_t val = (tmp1 > 0) ? tmp1 : huge;
      if (tmp2 > 0 && tmp2 < val) val = tmp2;
      return val;
    };
    raft::linalg::mapThenReduce(gamma, n_inactive, huge, map, cub::Min(),
                                stream, cor + n_active, a_vec);
  }
}

/**
 * @brief Initialize for Lars training.
 */
template <typename math_t, typename idx_t>
void larsInit(const raft::handle_t& handle, const math_t* X, idx_t n_rows,
              idx_t n_cols, idx_t ld_X, const math_t* y, math_t* Gram,
              idx_t ld_G, MLCommon::device_buffer<math_t>& U_buffer, math_t** U,
              idx_t* ld_U, MLCommon::host_buffer<idx_t>& indices,
              MLCommon::device_buffer<math_t>& cor,
              MLCommon::device_buffer<math_t>& sign, int* max_iter,
              math_t* coef_path, cudaStream_t stream) {
  if (n_cols < *max_iter) {
    *max_iter = n_cols;
  }
  if (Gram == nullptr) {
    const idx_t align_bytes = 256;
    *ld_U = raft::alignTo<idx_t>(*max_iter, align_bytes);
    // TODO Catch memory error, and recommend user to adjust max_iter
    // in a way that would avoid it
    U_buffer.resize((*ld_U) * (*max_iter), stream);
    *U = U_buffer.data();
  } else {
    // Set U as G. During the solution in larsFit, the Cholesky factorization
    // U will overwrite G.
    *U = Gram;
    *ld_U = ld_G;
  }
  std::iota(indices.data(), indices.data() + n_cols, 0);

  math_t one = 1;
  math_t zero = 0;
  // Set initial correlation to X.T * y
  CUBLAS_CHECK(raft::linalg::cublasgemv(handle.get_cublas_handle(), CUBLAS_OP_T,
                                        n_rows, n_cols, &one, X, ld_X, y, 1,
                                        &zero, cor.data(), 1, stream));
  if (coef_path) {
    std::cout << "Max iter " << *max_iter << "\n";
    CUDA_CHECK(cudaMemsetAsync(
      coef_path, 0, sizeof(math_t) * (*max_iter) * (*max_iter), stream));
  }
}

/**
 * @brief Update regression coefficient and correlations
 *
 *
 */
template <typename math_t, typename idx_t>
void updateCoef(const raft::handle_t& handle, idx_t max_iter, idx_t n_cols,
                idx_t n_active, math_t* gamma, const math_t* ws, math_t* cor,
                math_t* a_vec, math_t* beta, math_t* coef_path,
                MLCommon::device_buffer<math_t>& workspace,
                cudaStream_t stream) {
  // It is sufficient to update correlations only for the inactive set.
  // cor[n_active:] -= gamma * a_vec
  int n_inactive = n_cols - n_active;
  if (n_inactive > 0) {
    // We are accessing elements starting from cor + n_active. This might not be
    // aligned, therefore we copy the data to aligned memory loc.
    // TODO fix binary op to choose veclen automatically such that we do not
    // have alignment issues.
    raft::copy(workspace.data(), cor + n_active, n_inactive, stream);
    raft::linalg::binaryOp(
      workspace.data(), workspace.data(), a_vec, n_inactive,
      [gamma] __device__(math_t c, math_t a) { return c - *gamma * a; },
      stream);
    raft::copy(cor + n_active, workspace.data(), n_inactive, stream);
  }
  // beta[:n_active] += gamma * a
  raft::linalg::binaryOp(
    beta, beta, ws, n_active,
    [gamma] __device__(math_t b, math_t w) { return b + *gamma * w; }, stream);
  if (coef_path) {
    int i = n_active - 1;
    raft::copy(coef_path + i * max_iter, beta, n_active, stream);
  }
}

/**
 * @brief Train a regressor using Least Angre Regression.
 *
 * Least Angle Regression (LAR or LARS) is a model selection algorithm. It
 * builds up the model using the following algorithm:
 *
 * 1. We start with all the coefficients equal to zero.
 * 2. At each step we select the predictor that has the largest absolute
 *      correlation with the residual.
 * 3. We take the largest step possible in the direction which is equiangular
 *    with all the predictors selected so far. The largest step is determined
 *    such that using this step a new predictor will have as much correlation
 *    with the residual as any of the currently active predictors.
 * 4. Stop if max_iter reached or all the predictors are used, or if the
 *    correlation between any unused predictor and the residual is lower than
 *    a tolerance.
 *
 * The solver is based on [1]. The equations referred in the comments correspond
 * to the equations in the paper.
 *
 * Note: this algorithm assumes that the offset is removed from X and y, and
 * each feature is normalized:
 * - sum_i y_i = 0,
 * - sum_i x_{i,j} = 0, sum_i x_{i,j}^2=1 for j=0..n_col-1
 *
 * References:
 * [1] B. Efron, T. Hastie, I. Johnstone, R Tibshirani, Least Angle Regression
 * The Annals of Statistics (2004) Vol 32, No 2, 407-499
 * http://statweb.stanford.edu/~tibs/ftp/lars.pdf
 *
 * @param handle RAFT handle
 * @param X device array of training vectors in column major format,
 *     size [n_rows * n_cols]. Note that the columns of X will be permuted if
 *     the Gram matrix is not specified. It is expected that X is normalized so
 *     that each column has zero mean and unit variance.
 * @param n_rows number of training samples
 * @param n_cols number of feature columns
 * @param y device array of the regression targets, size [n_rows]. y should
 *     be normalized to have zero mean.
 * @param beta device array of regression coefficients, has to be allocated on
 *     entry, size [max_iter]
 * @param active_idx, device array containing the indices of active variables.
 *     Must be allocated on entry. Size [max_iter]
 * @param alphas device array to return the maximum correlation along the
 *     regularization path. Must be allocated on entry, size [max_iter+1].
 * @param n_active host pointer to return the number of active elements (scalar)
 * @param Gram device array containing Gram matrix containing X.T * X. Can be
 *     nullptr.
 * @param max_iter maximum number of iterations, this equals with the maximum
 *    number of coefficients returned. max_iter <= n_cols.
 * @param coef_path coefficients along the regularization path are returned
 *    here. Must be nullptr, or a device array already allocated on entry.
 *    Size [max_iter * max_iter].
 * @param verbosity verbosity level
 * @param ld_X leading dimension of X (stride of columns)
 * @param ld_G leading dimesion of G
 */
template <typename math_t, typename idx_t>
void larsFit(const raft::handle_t& handle, math_t* X, idx_t n_rows,
             idx_t n_cols, const math_t* y, math_t* beta, idx_t* active_idx,
             math_t* alphas, idx_t* n_active, math_t* Gram, int max_iter,
             math_t* coef_path, int verbosity, idx_t ld_X, idx_t ld_G) {
  ASSERT(n_cols > 0,
         "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 0,
         "Parameter n_rows: number of rows cannot be less than one");
  ML::Logger::get().setLevel(verbosity);

  // Set default ld parameters if needed.
  if (ld_X == 0) ld_X = n_rows;
  if (Gram && ld_G == 0) ld_G = n_cols;

  cudaStream_t stream = handle.get_stream();
  auto allocator = handle.get_device_allocator();

  // We will use either U_buffer.data() to store the Cholesky factorization, or
  // store it in place at Gram. Pointer U will point to the actual storage.
  MLCommon::device_buffer<math_t> U_buffer(allocator, stream);
  idx_t ld_U = 0;
  math_t* U = nullptr;

  // Indices of elements in the active set.
  MLCommon::host_buffer<idx_t> indices(handle.get_host_allocator(), stream,
                                       n_cols);
  // Sign of the correlation at the time when the element was added to the
  // active set.
  MLCommon::device_buffer<math_t> sign(allocator, stream, n_cols);

  // Correlation between the residual mu = y - X.T*beta and columns of X
  MLCommon::device_buffer<math_t> cor(allocator, stream, n_cols);

  // Temporary arrays used by the solver
  MLCommon::device_buffer<math_t> A(allocator, stream, 1);
  MLCommon::device_buffer<math_t> a_vec(allocator, stream, n_cols);
  MLCommon::device_buffer<math_t> gamma(allocator, stream, 1);
  MLCommon::device_buffer<math_t> u_eq(allocator, stream, n_rows);
  MLCommon::device_buffer<math_t> ws(allocator, stream, max_iter);
  MLCommon::device_buffer<math_t> workspace(allocator, stream, n_cols);

  larsInit(handle, X, n_rows, n_cols, ld_X, y, Gram, ld_G, U_buffer, &U, &ld_U,
           indices, cor, sign, &max_iter, coef_path, stream);

  math_t tolerance = std::numeric_limits<math_t>::epsilon();

  *n_active = 0;
  for (int i = 0; i < max_iter; i++) {
    math_t cj;
    idx_t j = selectMostCorrelated(*n_active, n_cols, cor.data(), &cj,
                                   workspace, stream);

    CUML_LOG_DEBUG("Iteration %d, cor=%f", i, cj);

    moveToActive(handle.get_cublas_handle(), n_active, j, X, n_rows, n_cols,
                 ld_X, cor.data(), indices.data(), Gram, ld_G, sign.data(),
                 stream);

    // safety check
    if (abs(cj) / n_rows < tolerance) {
      CUML_LOG_WARN("Iteration %d, reached tolarence limit  with %f", i,
                    abs(cj));
      break;
    }

    calcEquiangularVec(handle, *n_active, X, n_rows, n_cols, ld_X, sign.data(),
                       U, ld_U, Gram, ld_G, workspace, ws.data(), A.data(),
                       u_eq.data(), stream);

    calcMaxStep(handle, max_iter, n_rows, n_cols, *n_active, cj, A.data(),
                cor.data(), Gram, ld_G, X, ld_X, u_eq.data(), ws.data(),
                gamma.data(), a_vec.data(), stream);

    updateCoef(handle, max_iter, n_cols, *n_active, gamma.data(), ws.data(),
               cor.data(), a_vec.data(), beta, coef_path, workspace, stream);
  }

  // TODO apply sklearn definition of alphas
  raft::copy(alphas, cor.data(), *n_active, stream);
  CUDA_CHECK(cudaMemsetAsync(alphas + *n_active, 0, sizeof(math_t), stream));

  raft::update_device(active_idx, indices.data(), *n_active, stream);
}

/**
 * @brief Predict with least angle regressor.
 *
 * @param handle RAFT handle
 * @param X device array of training vectors in column major format,
 *     size [n_rows * n_cols].
 * @param n_rows number of training samples
 * @param n_cols number of feature columns
 * @param ld_X leading dimension of X (stride of columns)
 * @param beta device array of regression coefficients, size [n_active]
 * @param n_active the number of regression coefficients
 * @param active_idx, device array containing the indices of active variables.
 *     Only these columns of X will be used for prediction, size [n_active].
 * @param intercept
 * @param preds device array to store the predictions, size [n_rows]. Must be
 *     allocated on entry.
 */
template <typename math_t, typename idx_t>
void larsPredict(const raft::handle_t& handle, const math_t* X, idx_t n_rows,
                 idx_t n_cols, idx_t ld_X, const math_t* beta, idx_t n_active,
                 idx_t* active_idx, math_t intercept, math_t* preds) {
  cudaStream_t stream = handle.get_stream();
  auto allocator = handle.get_device_allocator();
  MLCommon::device_buffer<math_t> beta_sorted(allocator, stream);
  MLCommon::device_buffer<math_t> X_active_cols(allocator, stream);
  auto execution_policy = ML::thrust_exec_policy(allocator, stream);

  if (n_active == n_cols) {
    // We make a copy of the beta coefs and sort them
    beta_sorted.resize(n_active, stream);
    MLCommon::device_buffer<idx_t> idx_sorted(allocator, stream, n_active);
    raft::copy(beta_sorted.data(), beta, n_active, stream);
    raft::copy(idx_sorted.data(), active_idx, n_active, stream);
    thrust::device_ptr<math_t> beta_ptr(beta_sorted.data());
    thrust::device_ptr<idx_t> idx_ptr(idx_sorted.data());
    thrust::sort_by_key(execution_policy->on(stream), idx_ptr,
                        idx_ptr + n_active, beta_ptr);
    beta = beta_sorted.data();
  } else {
    // We collect active columns of X to contiguous space
    X_active_cols.resize(n_active * ld_X, stream);
    const int TPB = 64;
    MLCommon::Cache::get_vecs<<<raft::ceildiv(n_active, TPB), TPB, 0, stream>>>(
      X, ld_X, active_idx, n_active, X_active_cols.data());
    CUDA_CHECK(cudaPeekAtLastError());
    X = X_active_cols.data();
  }
  // Initialize preds = intercept
  thrust::device_ptr<math_t> pred_ptr(preds);
  thrust::fill(execution_policy->on(stream), pred_ptr, pred_ptr + n_rows,
               intercept);
  math_t one = 1;
  CUBLAS_CHECK(raft::linalg::cublasgemv(handle.get_cublas_handle(), CUBLAS_OP_N,
                                        n_rows, n_active, &one, X, ld_X, beta,
                                        1, &one, preds, 1, stream));
}
};  // namespace Lars
};  // namespace Solver
};  // end namespace ML
