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

#include <raft/cudart_utils.h>
#include <raft/linalg/add.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>
#include <rmm/device_uvector.hpp>
#include "preprocess.cuh"

namespace ML {
namespace GLM {

using namespace MLCommon;

template <typename math_t>
void ridgeSolve(const raft::handle_t &handle, math_t *S, math_t *V, math_t *U,
                int n_rows, int n_cols, math_t *b, math_t *alpha, int n_alpha,
                math_t *w, cudaStream_t stream) {
  auto cublasH = handle.get_cublas_handle();
  auto cusolverH = handle.get_cusolver_dn_handle();

  // Implements this: w = V * inv(S^2 + λ*I) * S * U^T * b
  rmm::device_uvector<math_t> S_nnz_vector(n_cols, stream);
  math_t *S_nnz = S_nnz_vector.data();
  math_t alp = math_t(1);
  math_t beta = math_t(0);
  math_t thres = math_t(1e-10);

  raft::matrix::setSmallValuesZero(S, n_cols, stream, thres);

  raft::copy(S_nnz, S, n_cols, stream);
  raft::matrix::power(S_nnz, n_cols, stream);
  raft::linalg::addScalar(S_nnz, S_nnz, alpha[0], n_cols, stream);
  raft::matrix::matrixVectorBinaryDivSkipZero(S, S_nnz, 1, n_cols, false, true,
                                              stream, true);

  raft::matrix::matrixVectorBinaryMult(V, S, n_cols, n_cols, false, true,
                                       stream);
  raft::linalg::gemm(handle, U, n_rows, n_cols, b, S_nnz, n_cols, 1,
                     CUBLAS_OP_T, CUBLAS_OP_N, alp, beta, stream);

  raft::linalg::gemm(handle, V, n_cols, n_cols, S_nnz, w, n_cols, 1,
                     CUBLAS_OP_N, CUBLAS_OP_N, alp, beta, stream);
}

template <typename math_t>
void ridgeSVD(const raft::handle_t &handle, math_t *A, int n_rows, int n_cols,
              math_t *b, math_t *alpha, int n_alpha, math_t *w,
              cudaStream_t stream) {
  auto cublasH = handle.get_cublas_handle();
  auto cusolverH = handle.get_cusolver_dn_handle();
  auto allocator = handle.get_device_allocator();

  ASSERT(n_cols > 0, "ridgeSVD: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "ridgeSVD: number of rows cannot be less than two");

  int U_len = n_rows * n_cols;
  int V_len = n_cols * n_cols;

  rmm::device_uvector<math_t> S(n_cols, stream);
  rmm::device_uvector<math_t> V(V_len, stream);
  rmm::device_uvector<math_t> U(U_len, stream);

  raft::linalg::svdQR(handle, A, n_rows, n_cols, S.data(), U.data(), V.data(),
                      true, true, true, stream);
  ridgeSolve(handle, S.data(), V.data(), U.data(), n_rows, n_cols, b, alpha,
             n_alpha, w, stream);
}

template <typename math_t>
void ridgeEig(const raft::handle_t &handle, math_t *A, int n_rows, int n_cols,
              math_t *b, math_t *alpha, int n_alpha, math_t *w,
              cudaStream_t stream) {
  auto cublasH = handle.get_cublas_handle();
  auto cusolverH = handle.get_cusolver_dn_handle();
  auto allocator = handle.get_device_allocator();

  ASSERT(n_cols > 1, "ridgeEig: number of columns cannot be less than two");
  ASSERT(n_rows > 1, "ridgeEig: number of rows cannot be less than two");

  int U_len = n_rows * n_cols;
  int V_len = n_cols * n_cols;

  rmm::device_uvector<math_t> S(n_cols, stream);
  rmm::device_uvector<math_t> V(V_len, stream);
  rmm::device_uvector<math_t> U(U_len, stream);

  raft::linalg::svdEig(handle, A, n_rows, n_cols, S.data(), U.data(), V.data(),
                       true, stream);

  ridgeSolve(handle, S.data(), V.data(), U.data(), n_rows, n_cols, b, alpha,
             n_alpha, w, stream);
}

/**
 * @brief fit a ridge regression model (l2 regularized least squares)
 * @param handle        cuml handle
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param labels        device pointer to label vector of length n_rows
 * @param alpha         device pointer to parameters of the l2 regularizer
 * @param n_alpha       number of regularization parameters
 * @param coef          device pointer to hold the solution for weights of size n_cols
 * @param intercept     device pointer to hold the solution for bias term of size 1
 * @param fit_intercept if true, fit intercept
 * @param normalize     if true, normalize data to zero mean, unit variance
 * @param stream        cuda stream
 * @param algo          specifies which solver to use (0: SVD, 1: Eigendecomposition)
 */
template <typename math_t>
void ridgeFit(const raft::handle_t &handle, math_t *input, int n_rows,
              int n_cols, math_t *labels, math_t *alpha, int n_alpha,
              math_t *coef, math_t *intercept, bool fit_intercept,
              bool normalize, cudaStream_t stream, int algo = 0) {
  auto cublas_handle = handle.get_cublas_handle();
  auto cusolver_handle = handle.get_cusolver_dn_handle();
  auto allocator = handle.get_device_allocator();

  ASSERT(n_cols > 0, "ridgeFit: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "ridgeFit: number of rows cannot be less than two");

  rmm::device_uvector<math_t> mu_input(0, stream);
  rmm::device_uvector<math_t> norm2_input(0, stream);
  rmm::device_uvector<math_t> mu_labels(0, stream);

  if (fit_intercept) {
    mu_input = rmm::device_uvector<math_t>(n_cols, stream);
    mu_labels = rmm::device_uvector<math_t>(1, stream);
    if (normalize) {
      norm2_input = rmm::device_uvector<math_t>(n_cols, stream);
    }
    preProcessData(handle, input, n_rows, n_cols, labels, intercept,
                   mu_input.data(), mu_labels.data(), norm2_input.data(),
                   fit_intercept, normalize, stream);
  }

  if (algo == 0 || n_cols == 1) {
    ridgeSVD(handle, input, n_rows, n_cols, labels, alpha, n_alpha, coef,
             stream);
  } else if (algo == 1) {
    ridgeEig(handle, input, n_rows, n_cols, labels, alpha, n_alpha, coef,
             stream);
  } else if (algo == 2) {
    ASSERT(false, "ridgeFit: no algorithm with this id has been implemented");
  } else {
    ASSERT(false, "ridgeFit: no algorithm with this id has been implemented");
  }

  if (fit_intercept) {
    postProcessData(handle, input, n_rows, n_cols, labels, coef, intercept,
                    mu_input.data(), mu_labels.data(), norm2_input.data(),
                    fit_intercept, normalize, stream);
  } else {
    *intercept = math_t(0);
  }
}

};  // namespace GLM
};  // namespace ML
// end namespace ML
