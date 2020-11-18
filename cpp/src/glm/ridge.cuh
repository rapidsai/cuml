/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
#include <common/cumlHandle.hpp>
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
  math_t *S_nnz;
  math_t alp = math_t(1);
  math_t beta = math_t(0);
  math_t thres = math_t(1e-10);

  raft::matrix::setSmallValuesZero(S, n_cols, stream, thres);
  raft::allocate(S_nnz, n_cols, true);
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

  CUDA_CHECK(cudaFree(S_nnz));
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

  math_t *S, *V, *U;

  int U_len = n_rows * n_cols;
  int V_len = n_cols * n_cols;

  raft::allocate(U, U_len);
  raft::allocate(V, V_len);
  raft::allocate(S, n_cols);

  raft::linalg::svdQR(handle, A, n_rows, n_cols, S, U, V, true, true, true,
                      stream);
  ridgeSolve(handle, S, V, U, n_rows, n_cols, b, alpha, n_alpha, w, stream);

  CUDA_CHECK(cudaFree(U));
  CUDA_CHECK(cudaFree(V));
  CUDA_CHECK(cudaFree(S));
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

  math_t *S, *V, *U;

  int U_len = n_rows * n_cols;
  int V_len = n_cols * n_cols;

  raft::allocate(U, U_len);
  raft::allocate(V, V_len);
  raft::allocate(S, n_cols);

  raft::linalg::svdEig(handle, A, n_rows, n_cols, S, U, V, true, stream);

  ridgeSolve(handle, S, V, U, n_rows, n_cols, b, alpha, n_alpha, w, stream);

  CUDA_CHECK(cudaFree(U));
  CUDA_CHECK(cudaFree(V));
  CUDA_CHECK(cudaFree(S));
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

  math_t *mu_input, *norm2_input, *mu_labels;

  if (fit_intercept) {
    raft::allocate(mu_input, n_cols);
    raft::allocate(mu_labels, 1);
    if (normalize) {
      raft::allocate(norm2_input, n_cols);
    }
    preProcessData(handle, input, n_rows, n_cols, labels, intercept, mu_input,
                   mu_labels, norm2_input, fit_intercept, normalize, stream);
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
                    mu_input, mu_labels, norm2_input, fit_intercept, normalize,
                    stream);

    if (normalize) {
      if (norm2_input != NULL) cudaFree(norm2_input);
    }

    if (mu_input != NULL) cudaFree(mu_input);
    if (mu_labels != NULL) cudaFree(mu_labels);
  } else {
    *intercept = math_t(0);
  }
}

/**
 * @brief to make predictions with a fitted ordinary least squares and ridge regression model
 * @param handle        cuml handle
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param coef          weights of the model
 * @param intercept     bias term of the model
 * @param preds         device pointer to store predictions of size n_rows
 * @param stream        cuda stream
 */
template <typename math_t>
void ridgePredict(const raft::handle_t &handle, const math_t *input, int n_rows,
                  int n_cols, const math_t *coef, math_t intercept,
                  math_t *preds, cudaStream_t stream) {
  ASSERT(n_cols > 0,
         "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1,
         "Parameter n_rows: number of rows cannot be less than two");

  math_t alpha = math_t(1);
  math_t beta = math_t(0);
  raft::linalg::gemm(handle, input, n_rows, n_cols, coef, preds, n_rows, 1,
                     CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, stream);

  raft::linalg::addScalar(preds, preds, intercept, n_rows, stream);
}

};  // namespace GLM
};  // namespace ML
// end namespace ML
