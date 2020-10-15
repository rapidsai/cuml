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
#include <raft/handle.hpp>
#include <raft/mr/device/buffer.hpp>
#include "eig.cuh"
#include "gemm.cuh"
#include "transpose.h"

namespace raft {
namespace linalg {

/**
 * @brief singular value decomposition (SVD) on the column major float type
 * input matrix using QR method
 * @param handle: raft handle
 * @param in: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param sing_vals: singular values of input matrix
 * @param left_sing_vecs: left singular values of input matrix
 * @param right_sing_vecs: right singular values of input matrix
 * @param trans_right: transpose right vectors or not
 * @param gen_left_vec: generate left eig vector. Not activated.
 * @param gen_right_vec: generate right eig vector. Not activated.
 * @param stream cuda stream
 */
// TODO: activate gen_left_vec and gen_right_vec options
// TODO: couldn't template this function due to cusolverDnSgesvd and
// cusolverSnSgesvd. Check if there is any other way.
template <typename T>
void svdQR(const raft::handle_t &handle, T *in, int n_rows, int n_cols,
           T *sing_vals, T *left_sing_vecs, T *right_sing_vecs,
           bool trans_right, bool gen_left_vec, bool gen_right_vec,
           cudaStream_t stream) {
  std::shared_ptr<raft::mr::device::allocator> allocator =
    handle.get_device_allocator();
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();
  cublasHandle_t cublasH = handle.get_cublas_handle();

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

  raft::mr::device::buffer<int> devInfo(allocator, stream, 1);
  T *d_rwork = nullptr;

  int lwork = 0;
  CUSOLVER_CHECK(
    cusolverDngesvd_bufferSize<T>(cusolverH, n_rows, n_cols, &lwork));
  raft::mr::device::buffer<T> d_work(allocator, stream, lwork);

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

  CUSOLVER_CHECK(cusolverDngesvd(
    cusolverH, jobu, jobvt, m, n, in, m, sing_vals, left_sing_vecs, m,
    right_sing_vecs, n, d_work.data(), lwork, d_rwork, devInfo.data(), stream));

  // Transpose the right singular vector back
  if (trans_right) raft::linalg::transpose(right_sing_vecs, n_cols, stream);

  CUDA_CHECK(cudaGetLastError());

  int dev_info;
  raft::update_host(&dev_info, devInfo.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT(dev_info == 0,
         "svd.cuh: svd couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");
}

template <typename T>
void svdEig(const raft::handle_t &handle, T *in, int n_rows, int n_cols, T *S,
            T *U, T *V, bool gen_left_vec, cudaStream_t stream) {
  auto allocator = handle.get_device_allocator();
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();
  cublasHandle_t cublasH = handle.get_cublas_handle();

  int len = n_cols * n_cols;
  raft::mr::device::buffer<T> in_cross_mult(allocator, stream, len);

  T alpha = T(1);
  T beta = T(0);
  raft::linalg::gemm(handle, in, n_rows, n_cols, in, in_cross_mult.data(),
                     n_cols, n_cols, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta,
                     stream);

  eigDC(handle, in_cross_mult.data(), n_cols, n_cols, V, S, stream);

  raft::matrix::colReverse(V, n_cols, n_cols, stream);
  raft::matrix::rowReverse(S, n_cols, 1, stream);

  raft::matrix::seqRoot(S, S, alpha, n_cols, stream, true);

  if (gen_left_vec) {
    raft::linalg::gemm(handle, in, n_rows, n_cols, V, U, n_rows, n_cols,
                       CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, stream);
    raft::matrix::matrixVectorBinaryDivSkipZero(U, S, n_rows, n_cols, false,
                                                true, stream);
  }
}

/**
 * @brief on the column major input matrix using Jacobi method
 * @param handle: raft handle
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
 * @param stream cuda stream
 */
template <typename math_t>
void svdJacobi(const raft::handle_t &handle, math_t *in, int n_rows, int n_cols,
               math_t *sing_vals, math_t *left_sing_vecs,
               math_t *right_sing_vecs, bool gen_left_vec, bool gen_right_vec,
               math_t tol, int max_sweeps, cudaStream_t stream) {
  auto allocator = handle.get_device_allocator();
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

  gesvdjInfo_t gesvdj_params = NULL;

  CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

  int m = n_rows;
  int n = n_cols;

  raft::mr::device::buffer<int> devInfo(allocator, stream, 1);

  int lwork = 0;
  int econ = 1;

  CUSOLVER_CHECK(raft::linalg::cusolverDngesvdj_bufferSize(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, in, m, sing_vals,
    left_sing_vecs, m, right_sing_vecs, n, &lwork, gesvdj_params));

  raft::mr::device::buffer<math_t> d_work(allocator, stream, lwork);

  CUSOLVER_CHECK(raft::linalg::cusolverDngesvdj(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, in, m, sing_vals,
    left_sing_vecs, m, right_sing_vecs, n, d_work.data(), lwork, devInfo.data(),
    gesvdj_params, stream));

  CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param handle: raft handle
 * @param U: left singular vectors of size n_rows x k
 * @param S: square matrix with singular values on its diagonal, k x k
 * @param V: right singular vectors of size n_cols x k
 * @param out: reconstructed matrix to be returned
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values
 * @param stream cuda stream
 */
template <typename math_t>
void svdReconstruction(const raft::handle_t &handle, math_t *U, math_t *S,
                       math_t *V, math_t *out, int n_rows, int n_cols, int k,
                       cudaStream_t stream) {
  auto allocator = handle.get_device_allocator();

  const math_t alpha = 1.0, beta = 0.0;
  raft::mr::device::buffer<math_t> SVT(allocator, stream, k * n_cols);

  raft::linalg::gemm(handle, S, k, k, V, SVT.data(), k, n_cols, CUBLAS_OP_N,
                     CUBLAS_OP_T, alpha, beta, stream);
  raft::linalg::gemm(handle, U, n_rows, k, SVT.data(), out, n_rows, n_cols,
                     CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, stream);
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param handle: raft handle
 * @param A_d: input matrix
 * @param U: left singular vectors of size n_rows x k
 * @param S_vec: singular values as a vector
 * @param V: right singular vectors of size n_cols x k
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values to be computed, 1.0 for normal SVD
 * @param tol: tolerance for the evaluation
 * @param stream cuda stream
 */
template <typename math_t>
bool evaluateSVDByL2Norm(const raft::handle_t &handle, math_t *A_d, math_t *U,
                         math_t *S_vec, math_t *V, int n_rows, int n_cols,
                         int k, math_t tol, cudaStream_t stream) {
  auto allocator = handle.get_device_allocator();
  cublasHandle_t cublasH = handle.get_cublas_handle();

  int m = n_rows, n = n_cols;

  // form product matrix
  raft::mr::device::buffer<math_t> P_d(allocator, stream, m * n);
  raft::mr::device::buffer<math_t> S_mat(allocator, stream, k * k);
  CUDA_CHECK(cudaMemsetAsync(P_d.data(), 0, sizeof(math_t) * m * n, stream));
  CUDA_CHECK(cudaMemsetAsync(S_mat.data(), 0, sizeof(math_t) * k * k, stream));

  raft::matrix::initializeDiagonalMatrix(S_vec, S_mat.data(), k, k, stream);
  svdReconstruction(handle, U, S_mat.data(), V, P_d.data(), m, n, k, stream);

  // get norms of each
  math_t normA = raft::matrix::getL2Norm(handle, A_d, m * n, stream);
  math_t normU = raft::matrix::getL2Norm(handle, U, m * k, stream);
  math_t normS = raft::matrix::getL2Norm(handle, S_mat.data(), k * k, stream);
  math_t normV = raft::matrix::getL2Norm(handle, V, n * k, stream);
  math_t normP = raft::matrix::getL2Norm(handle, P_d.data(), m * n, stream);

  // calculate percent error
  const math_t alpha = 1.0, beta = -1.0;
  raft::mr::device::buffer<math_t> A_minus_P(allocator, stream, m * n);
  CUDA_CHECK(
    cudaMemsetAsync(A_minus_P.data(), 0, sizeof(math_t) * m * n, stream));

  CUBLAS_CHECK(raft::linalg::cublasgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
                                        &alpha, A_d, m, &beta, P_d.data(), m,
                                        A_minus_P.data(), m, stream));

  math_t norm_A_minus_P =
    raft::matrix::getL2Norm(handle, A_minus_P.data(), m * n, stream);
  math_t percent_error = 100.0 * norm_A_minus_P / normA;
  return (percent_error / 100.0 < tol);
}

};  // end namespace linalg
};  // end namespace raft
