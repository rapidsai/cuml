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

#include <linalg/add.h>
#include <linalg/gemm.h>
#include <linalg/norm.h>
#include <linalg/subtract.h>
#include <linalg/svd.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <stats/stddev.h>
#include <stats/sum.h>
#include "common/cumlHandle.hpp"
#include "ml_utils.h"
#include "preprocess.h"

namespace ML {
namespace GLM {

using namespace MLCommon;

template <typename math_t>
void ridgeSolve(const cumlHandle_impl &handle, math_t *S, math_t *V, math_t *U,
                int n_rows, int n_cols, math_t *b, math_t *alpha, int n_alpha,
                math_t *w, cudaStream_t stream) {
  auto cublasH = handle.getCublasHandle();
  auto cusolverH = handle.getcusolverDnHandle();

  // Implements this: w = V * inv(S^2 + λ*I) * S * U^T * b
  math_t *S_nnz;
  math_t alp = math_t(1);
  math_t beta = math_t(0);
  math_t thres = math_t(1e-10);

  Matrix::setSmallValuesZero(S, n_cols, stream, thres);
  allocate(S_nnz, n_cols, true);
  copy(S_nnz, S, n_cols, stream);
  Matrix::power(S_nnz, n_cols, stream);
  LinAlg::addScalar(S_nnz, S_nnz, alpha[0], n_cols, stream);
  Matrix::matrixVectorBinaryDivSkipZero(S, S_nnz, 1, n_cols, false, true,
                                        stream, true);

  Matrix::matrixVectorBinaryMult(V, S, n_cols, n_cols, false, true, stream);
  LinAlg::gemm(U, n_rows, n_cols, b, S_nnz, n_cols, 1, CUBLAS_OP_T, CUBLAS_OP_N,
               alp, beta, cublasH, stream);

  LinAlg::gemm(V, n_cols, n_cols, S_nnz, w, n_cols, 1, CUBLAS_OP_N, CUBLAS_OP_N,
               alp, beta, cublasH, stream);

  CUDA_CHECK(cudaFree(S_nnz));
}

template <typename math_t>
void ridgeSVD(const cumlHandle_impl &handle, math_t *A, int n_rows, int n_cols,
              math_t *b, math_t *alpha, int n_alpha, math_t *w,
              cudaStream_t stream) {
  auto cublasH = handle.getCublasHandle();
  auto cusolverH = handle.getcusolverDnHandle();
  auto allocator = handle.getDeviceAllocator();

  ASSERT(n_cols > 0, "ridgeSVD: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "ridgeSVD: number of rows cannot be less than two");

  math_t *S, *V, *U;

  int U_len = n_rows * n_cols;
  int V_len = n_cols * n_cols;

  allocate(U, U_len);
  allocate(V, V_len);
  allocate(S, n_cols);

  LinAlg::svdQR(A, n_rows, n_cols, S, U, V, true, true, true, cusolverH,
                cublasH, allocator, stream);
  ridgeSolve(handle, S, V, U, n_rows, n_cols, b, alpha, n_alpha, w, stream);

  CUDA_CHECK(cudaFree(U));
  CUDA_CHECK(cudaFree(V));
  CUDA_CHECK(cudaFree(S));
}

template <typename math_t>
void ridgeEig(const cumlHandle_impl &handle, math_t *A, int n_rows, int n_cols,
              math_t *b, math_t *alpha, int n_alpha, math_t *w,
              cudaStream_t stream) {
  auto cublasH = handle.getCublasHandle();
  auto cusolverH = handle.getcusolverDnHandle();
  auto allocator = handle.getDeviceAllocator();

  ASSERT(n_cols > 1, "ridgeEig: number of columns cannot be less than two");
  ASSERT(n_rows > 1, "ridgeEig: number of rows cannot be less than two");

  math_t *S, *V, *U;

  int U_len = n_rows * n_cols;
  int V_len = n_cols * n_cols;

  allocate(U, U_len);
  allocate(V, V_len);
  allocate(S, n_cols);

  LinAlg::svdEig(A, n_rows, n_cols, S, U, V, true, cublasH, cusolverH, stream,
                 allocator);
  ridgeSolve(handle, S, V, U, n_rows, n_cols, b, alpha, n_alpha, w, stream);

  CUDA_CHECK(cudaFree(U));
  CUDA_CHECK(cudaFree(V));
  CUDA_CHECK(cudaFree(S));
}

/**
 * @defgroup Functions fit a ridge regression model (l2 regularized least squares)
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
 * @param algo          specifies which solver to use (0: SVD, 1: Eigendecomposition)
 * @{
 */
template <typename math_t>
void ridgeFit(const cumlHandle_impl &handle, math_t *input, int n_rows,
              int n_cols, math_t *labels, math_t *alpha, int n_alpha,
              math_t *coef, math_t *intercept, bool fit_intercept,
              bool normalize, cudaStream_t stream, int algo = 0) {
  auto cublas_handle = handle.getCublasHandle();
  auto cusolver_handle = handle.getcusolverDnHandle();
  auto allocator = handle.getDeviceAllocator();

  ASSERT(n_cols > 0, "ridgeFit: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "ridgeFit: number of rows cannot be less than two");

  math_t *mu_input, *norm2_input, *mu_labels;

  if (fit_intercept) {
    allocate(mu_input, n_cols);
    allocate(mu_labels, 1);
    if (normalize) {
      allocate(norm2_input, n_cols);
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
 * @defgroup Functions to make predictions with a fitted ordinary least squares and ridge regression model
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param coef          weights of the model
 * @param intercept     bias term of the model
 * @param preds         device pointer to store predictions of size n_rows
 * @{
 */
template <typename math_t>
void ridgePredict(const cumlHandle_impl &handle, const math_t *input,
                  int n_rows, int n_cols, const math_t *coef, math_t intercept,
                  math_t *preds, cudaStream_t stream) {
  auto cublas_handle = handle.getCublasHandle();

  ASSERT(n_cols > 0,
         "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1,
         "Parameter n_rows: number of rows cannot be less than two");

  math_t alpha = math_t(1);
  math_t beta = math_t(0);
  LinAlg::gemm(input, n_rows, n_cols, coef, preds, n_rows, 1, CUBLAS_OP_N,
               CUBLAS_OP_N, alpha, beta, cublas_handle, stream);

  LinAlg::addScalar(preds, preds, intercept, n_rows, stream);
}

/** @} */
};  // namespace GLM
};  // namespace ML
// end namespace ML
