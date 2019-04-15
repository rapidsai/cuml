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

#include "../matrix/math.h"
#include "../matrix/matrix.h"
#include "../random/rng.h"
#include "cublas_wrappers.h"
#include "cuda_utils.h"
#include "cusolver_wrappers.h"
#include "eig.h"
#include "gemm.h"
#include "qr.h"
#include "svd.h"
#include "transpose.h"

namespace MLCommon {
namespace LinAlg {


/**
 * @defgroup randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying no. of PCs and
 * upsamples directly
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param k: no. of singular values to be computed
 * @param p: no. of upsamples
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param cusolverH cusolver handle
 * @param cublasH cublas handle
 * @param allocator device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void rsvdFixedRank(math_t *M, int n_rows, int n_cols, math_t *&S_vec,
                   math_t *&U, math_t *&V, int k, int p, bool use_bbt,
                   bool gen_left_vec, bool gen_right_vec, bool use_jacobi,
                   math_t tol, int max_sweeps, cusolverDnHandle_t cusolverH,
                   cublasHandle_t cublasH, cudaStream_t stream,
                   std::shared_ptr<deviceAllocator> allocator) {
  // All the notations are following Algorithm 4 & 5 in S. Voronin's paper:
  // https://arxiv.org/abs/1502.05366

  int m = n_rows, n = n_cols;
  int l =
    k + p;   // Total number of singular values to be computed before truncation
  int q = 2; // Number of power sampling counts
  int s = 1; // Frequency controller for QR decomposition during power sampling
             // scheme. s = 1: 2 QR per iteration; s = 2: 1 QR per iteration; s
             // > 2: less frequent QR

  const math_t alpha = 1.0, beta = 0.0;

  // Build temporary U, S, V matrices
  device_buffer<math_t> S_vec_tmp(allocator, stream, l);
  CUDA_CHECK(cudaMemsetAsync(S_vec_tmp.data(), 0, sizeof(math_t) * l, stream));

  // build random matrix
  device_buffer<math_t> RN(allocator, stream, n * l);
  Random::Rng rng(484);
  rng.normal(RN.data(), n * l, math_t(0.0), alpha, stream);

  // multiply to get matrix of random samples Y
  device_buffer<math_t> Y(allocator, stream, m * l);
  gemm(M, m, n, RN.data(), Y.data(), m, l, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, cublasH, stream);

  // now build up (M M^T)^q R
  device_buffer<math_t> Z(allocator, stream, n * l);
  CUDA_CHECK(cudaMemsetAsync(Z.data(), 0, sizeof(math_t) * n * l, stream));
  device_buffer<math_t> Yorth(allocator, stream, m * l);
  CUDA_CHECK(cudaMemsetAsync(Yorth.data(), 0, sizeof(math_t) * m * l, stream));
  device_buffer<math_t> Zorth(allocator, stream, n * l);
  CUDA_CHECK(cudaMemsetAsync(Zorth.data(), 0, sizeof(math_t) * n * l, stream));

  // power sampling scheme
  for (int j = 1; j < q; j++) {
    if ((2 * j - 2) % s == 0) {
      qrGetQ(Y.data(), Yorth.data(), m, l, cusolverH, stream, allocator);
      gemm(M, m, n, Yorth.data(), Z.data(), n, l, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta,
           cublasH, stream);
    } else {
      gemm(M, m, n, Y.data(), Z.data(), n, l, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, cublasH, stream);
    }

    if ((2 * j - 1) % s == 0) {
      qrGetQ(Z.data(), Zorth.data(), n, l, cusolverH, stream, allocator);
      gemm(M, m, n, Zorth.data(), Y.data(), m, l, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta,
           cublasH, stream);
    } else {
      gemm(M, m, n, Z.data(), Y.data(), m, l, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, cublasH, stream);
    }
  }

  // orthogonalize on exit from loop to get Q
  device_buffer<math_t> Q(allocator, stream, m * l);
  CUDA_CHECK(cudaMemsetAsync(Q.data(), 0, sizeof(math_t) * m * l, stream));
  qrGetQ(Y.data(), Q.data(), m, l, cusolverH, stream, allocator);

  // either QR of B^T method, or eigendecompose BB^T method
  if (!use_bbt) {
    // form Bt = Mt*Q : nxm * mxl = nxl
    device_buffer<math_t> Bt(allocator, stream, n * l);
    CUDA_CHECK(cudaMemsetAsync(Bt.data(), 0, sizeof(math_t) * n * l, stream));
    gemm(M, m, n, Q.data(), Bt.data(), n, l, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, cublasH, stream);

    // compute QR factorization of Bt
    // M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */
    device_buffer<math_t> Qhat(allocator, stream, n * l);
    CUDA_CHECK(cudaMemsetAsync(Qhat.data(), 0, sizeof(math_t) * n * l, stream));
    device_buffer<math_t> Rhat(allocator, stream, l * l);
    CUDA_CHECK(cudaMemsetAsync(Rhat.data(), 0, sizeof(math_t) * l * l, stream));
    qrGetQR(Bt.data(), Qhat.data(), Rhat.data(), n, l, cusolverH, stream, allocator);

    // compute SVD of Rhat (lxl)
    device_buffer<math_t> Uhat(allocator, stream, l * l);
    CUDA_CHECK(cudaMemsetAsync(Uhat.data(), 0, sizeof(math_t) * l * l, stream));
    device_buffer<math_t> Vhat(allocator, stream, l * l);
    CUDA_CHECK(cudaMemsetAsync(Vhat.data(), 0, sizeof(math_t) * l * l, stream));
    if (use_jacobi)
      svdJacobi(Rhat.data(), l, l, S_vec_tmp.data(), Uhat.data(), Vhat.data(), true, true, tol, max_sweeps,
                cusolverH, stream, allocator);
    else
      svdQR(Rhat.data(), l, l, S_vec_tmp.data(), Uhat.data(), Vhat.data(), true, true, true, cusolverH, cublasH,
            allocator, stream);
    Matrix::sliceMatrix(S_vec_tmp.data(), 1, l, S_vec, 0, 0, 1,
                        k, stream); // First k elements of S_vec

    // Merge step 14 & 15 by calculating U = Q*Vhat[:,1:k] mxl * lxk = mxk
    if (gen_left_vec) {
      gemm(Q.data(), m, l, Vhat.data(), U, m, k /*used to be l and needs slicing*/,
           CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, cublasH, stream);
    }

    // Merge step 14 & 15 by calculating V = Qhat*Uhat[:,1:k] nxl * lxk = nxk
    if (gen_right_vec) {
      gemm(Qhat.data(), n, l, Uhat.data(), V, n, k /*used to be l and needs slicing*/,
           CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, cublasH, stream);
    }
  } else {
    // build the matrix B B^T = Q^T M M^T Q column by column
    // Bt = M^T Q ; nxm * mxk = nxk
    device_buffer<math_t> B(allocator, stream, n * l);
    gemm(Q.data(), m, l, M, B.data(), l, n, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, cublasH, stream);

    device_buffer<math_t> BBt(allocator, stream, l * l);
    gemm(B.data(), l, n, B.data(), BBt.data(), l, l, CUBLAS_OP_N, CUBLAS_OP_T, alpha, beta, cublasH, stream);

    // compute eigendecomposition of BBt
    device_buffer<math_t> Uhat(allocator, stream, l * l);
    CUDA_CHECK(cudaMemsetAsync(Uhat.data(), 0, sizeof(math_t) * l * l, stream));
    device_buffer<math_t> Uhat_dup(allocator, stream, l * l);
    CUDA_CHECK(cudaMemsetAsync(Uhat_dup.data(), 0, sizeof(math_t) * l * l, stream));
    Matrix::copyUpperTriangular(BBt.data(), Uhat_dup.data(), l, l, stream);
    if (use_jacobi)
      eigJacobi(Uhat_dup.data(), l, l, Uhat.data(), S_vec_tmp.data(), tol, max_sweeps, cusolverH,
                stream, allocator);
    else
      eigDC(Uhat_dup.data(), l, l, Uhat.data(), S_vec_tmp.data(), cusolverH, stream, allocator);
    Matrix::seqRoot(S_vec_tmp.data(), l, stream);
    Matrix::sliceMatrix(S_vec_tmp.data(), 1, l, S_vec, 0, p, 1,
                        l, stream); // Last k elements of S_vec
    Matrix::colReverse(S_vec, 1, k, stream);

    // Merge step 14 & 15 by calculating U = Q*Uhat[:,(p+1):l] mxl * lxk = mxk
    if (gen_left_vec) {
      gemm(Q.data(), m, l, Uhat.data() + p * l, U, m, k, CUBLAS_OP_N, CUBLAS_OP_N, alpha,
           beta, cublasH, stream);
      Matrix::colReverse(U, m, k, stream);
    }

    // Merge step 14 & 15 by calculating V = B^T Uhat[:,(p+1):l] *
    // Sigma^{-1}[(p+1):l, (p+1):l] nxl * lxk * kxk = nxk
    if (gen_right_vec) {
      device_buffer<math_t> Sinv(allocator, stream, k * k);
      CUDA_CHECK(cudaMemsetAsync(Sinv.data(), 0, sizeof(math_t) * k * k, stream));
      device_buffer<math_t> UhatSinv(allocator, stream, l * k);
      CUDA_CHECK(cudaMemsetAsync(UhatSinv.data(), 0, sizeof(math_t) * l * k, stream));
      Matrix::reciprocal(S_vec_tmp.data(), l, stream);
      Matrix::initializeDiagonalMatrix(S_vec_tmp.data() + p, Sinv.data(), k, k, stream);

      gemm(Uhat.data() + p * l, l, k, Sinv.data(), UhatSinv.data(), l, k, CUBLAS_OP_N, CUBLAS_OP_N,
           alpha, beta, cublasH, stream);
      gemm(B.data(), l, n, UhatSinv.data(), V, n, k, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta,
            cublasH, stream);
      Matrix::colReverse(V, n, k, stream);
    }
  }
}

/**
 * @defgroup randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying the PC and upsampling
 * ratio
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param PC_perc: percentage of singular values to be computed
 * @param UpS_perc: upsampling percentage
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param cusolverH cusolver handle
 * @param cublasH cublas handle
 * @param stream cuda stream
 * @param allocator device allocator for temporary buffers during computation
 */
template <typename math_t>
void rsvdPerc(math_t *M, int n_rows, int n_cols, math_t *&S_vec, math_t *&U,
              math_t *&V, math_t PC_perc, math_t UpS_perc, bool use_bbt,
              bool gen_left_vec, bool gen_right_vec, bool use_jacobi,
              math_t tol, int max_sweeps, cusolverDnHandle_t cusolverH,
              cublasHandle_t cublasH, cudaStream_t stream,
              std::shared_ptr<deviceAllocator> allocator) {
  int k = max((int)(min(n_rows, n_cols) * PC_perc),
              1); // Number of singular values to be computed
  int p = max((int)(min(n_rows, n_cols) * UpS_perc), 1); // Upsamples
  rsvdFixedRank(M, n_rows, n_cols, S_vec, U, V, k, p, use_bbt, gen_left_vec,
                gen_right_vec, use_jacobi, tol, max_sweeps, cusolverH, cublasH,
                stream, allocator);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
