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

#include "cuda_utils.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/gemm.h"
#include "mean_center.h"

namespace MLCommon {
namespace Stats {

template <typename math_t>
__global__ static void mean_trick_kernel(const math_t *__restrict sum,
                                         const math_t *__restrict mu,
                                         const math_t multiplier, const int p,
                                         math_t *__restrict XTX_) {
// (X - mu).T @ (X - mu) == X.T @ X - X.T @ mu
#define XTX(i, j) XTX_[(i) + (j)*p]

  const int j = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every column
  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every row
  if (i > j or j >= p) return;  // Only process upper triangular

  XTX(i, j) -= sum[j] * mu[i];
  XTX(i, j) *= multiplier;  // For XTX / n
}

/**
 * @brief Compute covariance of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @param covar the output covariance matrix
 * @param data the input matrix (this will get mean-centered at the end!)
 * @param mu mean vector of the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample covariance or not. In other words,
 * whether to normalize the output using N-1 or N, for true or false,
 * respectively
 * @param rowMajor whether the input data is row or col major
 * @param stable whether to run the slower-but-numerically-stable version or not
 * @param handle cublas handle
 * @note if stable=true, then the input data will be mean centered after this
 * function returns!
 */
template <typename Type>
void cov(Type *covar, Type *data, const Type *mu, int D, int N, bool sample,
         bool rowMajor, bool stable, cublasHandle_t handle,
         cudaStream_t stream) {
  if (stable) {
    // since mean operation is assumed to be along a given column, broadcast
    // must be along rows!
    meanCenter(data, data, mu, D, N, rowMajor, true, stream);
    Type alpha = Type(1) / (sample ? Type(N - 1) : Type(N));
    Type beta = Type(0);
    if (rowMajor) {
      CUBLAS_CHECK(LinAlg::cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, D, D, N,
                                      &alpha, data, D, data, D, &beta, covar, D,
                                      stream));
    } else {
      LinAlg::gemm(data, N, D, data, covar, D, D, CUBLAS_OP_T, CUBLAS_OP_N,
                   alpha, beta, handle, stream);
    }
  } else {
    ///@todo: implement this using cutlass + customized epilogue!
    ASSERT(false, "cov: Implement stable=false case!");
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Compute covariance of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @param covar the output covariance matrix
 * @param data the input matrix (this will get mean-centered at the end! if dtype != float32 or float64)
 * @param mu column mean vector of the input matrix
 * @param sum column sum vector of the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample covariance or not. In other words,
 * whether to normalize the output using N-1 or N, for true or false,
 * respectively
 * @param rowMajor whether the input data is row or col major
 * @param stable whether to run the slower-but-numerically-stable version or not
 * @param handle cublas handle
 * @note if stable=true, then the input data will be mean centered after this
 * function returns!
 */
template <typename Type, int TPB_X = 32, int TPB_Y = 32>
void cov(Type *covar, Type *data, const Type *mu,
         const Type *sum,  // Fast path has column sum
         int D, int N, bool sample, bool rowMajor, bool stable,
         cublasHandle_t handle, cudaStream_t stream) {
  if (stable) {
    // since mean operation is assumed to be along a given column, broadcast
    // must be along rows!

    if (rowMajor) {
      meanCenter(data, data, mu, D, N, rowMajor, true, stream);
      Type alpha = Type(1) / (sample ? Type(N - 1) : Type(N));
      Type beta = Type(0);

      CUBLAS_CHECK(LinAlg::cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, D, D, N,
                                      &alpha, data, D, data, D, &beta, covar, D,
                                      stream));
    } else if ((sizeof(Type) != sizeof(float)) and
               (sizeof(Type) != sizeof(double))) {
      // Uses old in place centering
      meanCenter(data, data, mu, D, N, rowMajor, true, stream);
      Type alpha = Type(1) / (sample ? Type(N - 1) : Type(N));
      Type beta = Type(0);

      // Other data types
      LinAlg::gemm(data, N, D, data, covar, D, D, CUBLAS_OP_T, CUBLAS_OP_N,
                   alpha, beta, handle, stream);
    } else {
      //meanCenter(data, data, mu, D, N, rowMajor, true, stream);
      Type alpha = Type(1);
      Type beta = Type(0);

      LinAlg::gemm(data, N, D, data, covar, D, D, CUBLAS_OP_T, CUBLAS_OP_N,
                   alpha, beta, handle, stream);

      if (sizeof(Type) == sizeof(float)) {
        // Produces X.T @ X float32
        CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, D,
                                 N, (float *)&alpha, (float *)data, N,
                                 (float *)&beta, (float *)covar, D));
      } else if (sizeof(Type) == sizeof(double)) {
        // Produces X.T @ X float64
        CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, D,
                                 N, (double *)&alpha, (double *)data, N,
                                 (double *)&beta, (double *)covar, D));
      }

      const dim3 threadsPerBlock(TPB_X, TPB_Y);
      const dim3 numBlocks(MLCommon::ceildiv(D, TPB_X),
                           MLCommon::ceildiv(D, TPB_Y));

      // Divide up sampling ratio 1/n or 1/(n-1)
      Type divide = Type(1) / (sample ? Type(N - 1) : Type(N));

      // Instead of computing (X - mu).T @ (X - mu), use the mean removal trick
      // of X.T @ X - X.T @ mu
      mean_trick_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        sum, mu, divide, D, covar);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  } else {
    ///@todo: implement this using cutlass + customized epilogue!
    ASSERT(false, "cov: Implement stable=false case!");
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // end namespace Stats
};  // end namespace MLCommon
