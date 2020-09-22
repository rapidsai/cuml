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

#include <common/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <common/device_buffer.hpp>
#include <cuda_utils.cuh>
#include <cuml/common/cuml_allocator.hpp>
#include <matrix/math.cuh>
#include <matrix/matrix.cuh>
#include "eig.cuh"
#include "gemm.cuh"
#include "transpose.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @brief singular value decomposition (SVD) on the column major float type
 * input matrix using QR method
 * @param in: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param sing_vals: singular values of input matrix
 * @param left_sing_vecs: left singular values of input matrix
 * @param right_sing_vecs: right singular values of input matrix
 * @param trans_right: transpose right vectors or not
 * @param gen_left_vec: generate left eig vector. Not activated.
 * @param gen_right_vec: generate right eig vector. Not activated.
 * @param cusolverH cusolver handle
 * @param cublasH cublas handle
 * @param allocator device allocator for temporary buffers during computation
 * @param stream cuda stream
 */
// TODO: activate gen_left_vec and gen_right_vec options
// TODO: couldn't template this function due to cusolverDnSgesvd and
// cusolverSnSgesvd. Check if there is any other way.
template <typename T>
void svdQR(T *in, int n_rows, int n_cols, T *sing_vals, T *left_sing_vecs,
           T *right_sing_vecs, bool trans_right, bool gen_left_vec,
           bool gen_right_vec, cusolverDnHandle_t cusolverH,
           cublasHandle_t cublasH, std::shared_ptr<deviceAllocator> allocator,
           cudaStream_t stream) {
#if CUDART_VERSION >= 10010
  // 46340: sqrt of max int value
  ASSERT(n_rows <= 46340,
         "svd solver is not supported for the data that has more than 46340 "
         "samples (rows) "
         "if you are using CUDA version 10.1. Please use other solvers such as "
         "eig if it is available.");
#endif

  const int m = n_rows;
  const int n = n_cols;

  device_buffer<int> devInfo(allocator, stream, 1);
  T *d_rwork = nullptr;

  int lwork = 0;
  CUSOLVER_CHECK(raft::linalg::cusolverDngesvd_bufferSize<T>(cusolverH, n_rows,
                                                             n_cols, &lwork));
  device_buffer<T> d_work(allocator, stream, lwork);

  char jobu = 'S';
  char jobvt = 'A';

  if (!gen_left_vec) {
    char new_u = 'N';
    strcpy(&jobu, &new_u);
  }

  if (!gen_right_vec) {
    char new_vt = 'N';
    strcpy(&jobvt, &new_vt);
  }

  CUSOLVER_CHECK(raft::linalg::cusolverDngesvd(
    cusolverH, jobu, jobvt, m, n, in, m, sing_vals, left_sing_vecs, m,
    right_sing_vecs, n, d_work.data(), lwork, d_rwork, devInfo.data(), stream));

  // Transpose the right singular vector back
  if (trans_right) transpose(right_sing_vecs, n_cols, stream);

  CUDA_CHECK(cudaGetLastError());

  int dev_info;
  updateHost(&dev_info, devInfo.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT(dev_info == 0,
         "svd.cuh: svd couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");
}

template <typename T>
void svdEig(T *in, int n_rows, int n_cols, T *S, T *U, T *V, bool gen_left_vec,
            cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
            cudaStream_t stream, std::shared_ptr<deviceAllocator> allocator) {
  int len = n_cols * n_cols;
  device_buffer<T> in_cross_mult(allocator, stream, len);

  T alpha = T(1);
  T beta = T(0);
  gemm(in, n_rows, n_cols, in, in_cross_mult.data(), n_cols, n_cols,
       CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, cublasH, stream);

  eigDC(in_cross_mult.data(), n_cols, n_cols, V, S, cusolverH, stream,
        allocator);

  Matrix::colReverse(V, n_cols, n_cols, stream);
  Matrix::rowReverse(S, n_cols, 1, stream);

  Matrix::seqRoot(S, S, alpha, n_cols, stream, true);

  if (gen_left_vec) {
    gemm(in, n_rows, n_cols, V, U, n_rows, n_cols, CUBLAS_OP_N, CUBLAS_OP_N,
         alpha, beta, cublasH, stream);
    Matrix::matrixVectorBinaryDivSkipZero(U, S, n_rows, n_cols, false, true,
                                          stream);
  }
}

/**
 * @brief on the column major input matrix using Jacobi method
 * @param in: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param sing_vals: singular values of input matrix
 * @param left_sing_vecs: left singular vectors of input matrix
 * @param right_sing_vecs: right singular vectors of input matrix
 * @param gen_left_vec: generate left eig vector. Not activated.
 * @param gen_right_vec: generate right eig vector. Not activated.
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the
 * error is below tol
 * @param max_sweeps: number of sweeps in the Jacobi algorithm. The more the better
 * accuracy.
 * @param cusolverH cusolver handle
 * @param stream cuda stream
 * @param allocator device allocator for temporary buffers during computation
 */
template <typename math_t>
void svdJacobi(math_t *in, int n_rows, int n_cols, math_t *sing_vals,
               math_t *left_sing_vecs, math_t *right_sing_vecs,
               bool gen_left_vec, bool gen_right_vec, math_t tol,
               int max_sweeps, cusolverDnHandle_t cusolverH,
               cudaStream_t stream,
               std::shared_ptr<deviceAllocator> allocator) {
  gesvdjInfo_t gesvdj_params = NULL;

  CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

  int m = n_rows;
  int n = n_cols;

  device_buffer<int> devInfo(allocator, stream, 1);

  int lwork = 0;
  int econ = 1;

  CUSOLVER_CHECK(raft::linalg::cusolverDngesvdj_bufferSize(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, in, m, sing_vals,
    left_sing_vecs, m, right_sing_vecs, n, &lwork, gesvdj_params));

  device_buffer<math_t> d_work(allocator, stream, lwork);

  CUSOLVER_CHECK(raft::linalg::cusolverDngesvdj(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, in, m, sing_vals,
    left_sing_vecs, m, right_sing_vecs, n, d_work.data(), lwork, devInfo.data(),
    gesvdj_params, stream));

  CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param U: left singular vectors of size n_rows x k
 * @param S: square matrix with singular values on its diagonal, k x k
 * @param V: right singular vectors of size n_cols x k
 * @param out: reconstructed matrix to be returned
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values
 * @param cublasH cublas handle
 * @param stream cuda stream
 * @param allocator device allocator for temporary buffers during computation
 */
template <typename math_t>
void svdReconstruction(math_t *U, math_t *S, math_t *V, math_t *out, int n_rows,
                       int n_cols, int k, cublasHandle_t cublasH,
                       cudaStream_t stream,
                       std::shared_ptr<deviceAllocator> allocator) {
  const math_t alpha = 1.0, beta = 0.0;
  device_buffer<math_t> SVT(allocator, stream, k * n_cols);

  gemm(S, k, k, V, SVT.data(), k, n_cols, CUBLAS_OP_N, CUBLAS_OP_T, alpha, beta,
       cublasH, stream);
  gemm(U, n_rows, k, SVT.data(), out, n_rows, n_cols, CUBLAS_OP_N, CUBLAS_OP_N,
       alpha, beta, cublasH, stream);
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param A_d: input matrix
 * @param U: left singular vectors of size n_rows x k
 * @param S_vec: singular values as a vector
 * @param V: right singular vectors of size n_cols x k
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values to be computed, 1.0 for normal SVD
 * @param tol: tolerance for the evaluation
 * @param cublasH cublas handle
 * @param stream cuda stream
 * @param allocator device allocator for temporary buffers during computation
 */
template <typename math_t>
bool evaluateSVDByL2Norm(math_t *A_d, math_t *U, math_t *S_vec, math_t *V,
                         int n_rows, int n_cols, int k, math_t tol,
                         cublasHandle_t cublasH, cudaStream_t stream,
                         std::shared_ptr<deviceAllocator> allocator) {
  int m = n_rows, n = n_cols;

  // form product matrix
  device_buffer<math_t> P_d(allocator, stream, m * n);
  device_buffer<math_t> S_mat(allocator, stream, k * k);
  CUDA_CHECK(cudaMemsetAsync(P_d.data(), 0, sizeof(math_t) * m * n, stream));
  CUDA_CHECK(cudaMemsetAsync(S_mat.data(), 0, sizeof(math_t) * k * k, stream));

  Matrix::initializeDiagonalMatrix(S_vec, S_mat.data(), k, k, stream);
  svdReconstruction(U, S_mat.data(), V, P_d.data(), m, n, k, cublasH, stream,
                    allocator);

  // get norms of each
  math_t normA = Matrix::getL2Norm(A_d, m * n, cublasH, stream);
  math_t normU = Matrix::getL2Norm(U, m * k, cublasH, stream);
  math_t normS = Matrix::getL2Norm(S_mat.data(), k * k, cublasH, stream);
  math_t normV = Matrix::getL2Norm(V, n * k, cublasH, stream);
  math_t normP = Matrix::getL2Norm(P_d.data(), m * n, cublasH, stream);

  // calculate percent error
  const math_t alpha = 1.0, beta = -1.0;
  device_buffer<math_t> A_minus_P(allocator, stream, m * n);
  CUDA_CHECK(
    cudaMemsetAsync(A_minus_P.data(), 0, sizeof(math_t) * m * n, stream));

  CUBLAS_CHECK(raft::linalg::cublasgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
                                        &alpha, A_d, m, &beta, P_d.data(), m,
                                        A_minus_P.data(), m, stream));

  math_t norm_A_minus_P =
    Matrix::getL2Norm(A_minus_P.data(), m * n, cublasH, stream);
  math_t percent_error = 100.0 * norm_A_minus_P / normA;
  return (percent_error / 100.0 < tol);
}

};  // end namespace LinAlg
};  // end namespace MLCommon
