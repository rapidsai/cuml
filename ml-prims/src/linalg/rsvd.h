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
#include "device_allocator.h"
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
 * @param mgr device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void rsvdFixedRank(math_t *M, int n_rows, int n_cols, math_t *&S_vec,
                   math_t *&U, math_t *&V, int k, int p, bool use_bbt,
                   bool gen_left_vec, bool gen_right_vec, bool use_jacobi,
                   math_t tol, int max_sweeps, cusolverDnHandle_t cusolverH,
                   cublasHandle_t cublasH, DeviceAllocator &mgr) {
  // All the notations are following Algorithm 4 & 5 in S. Voronin's paper:
  // https://arxiv.org/abs/1502.05366

  ///@todo: what if stream in cusolver/cublas handles are different!?
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublasH, &stream));

  int m = n_rows, n = n_cols;
  int l =
    k + p;   // Total number of singular values to be computed before truncation
  int q = 2; // Number of power sampling counts
  int s = 1; // Frequency controller for QR decomposition during power sampling
             // scheme. s = 1: 2 QR per iteration; s = 2: 1 QR per iteration; s
             // > 2: less frequent QR

  const math_t alpha = 1.0, beta = 0.0;

  // Build temporary U, S, V matrices
  math_t *S_vec_tmp = (math_t *)mgr.alloc(sizeof(math_t) * l);
  CUDA_CHECK(cudaMemsetAsync(S_vec_tmp, 0, sizeof(math_t) * l, stream));

  // build random matrix
  math_t *RN = (math_t *)mgr.alloc(sizeof(math_t) * n * l);
  Random::Rng<math_t> rng(484);
  rng.normal(RN, n * l, 0.0, alpha, stream);

  // multiply to get matrix of random samples Y
  math_t *Y = (math_t *)mgr.alloc(sizeof(math_t) * m * l);
  gemm(M, m, n, RN, Y, m, l, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, cublasH);

  // now build up (M M^T)^q R
  math_t *Z = (math_t *)mgr.alloc(sizeof(math_t) * n * l);
  CUDA_CHECK(cudaMemsetAsync(Z, 0, sizeof(math_t) * n * l, stream));
  math_t *Yorth = (math_t *)mgr.alloc(sizeof(math_t) * m * l);
  CUDA_CHECK(cudaMemsetAsync(Yorth, 0, sizeof(math_t) * m * l, stream));
  math_t *Zorth = (math_t *)mgr.alloc(sizeof(math_t) * n * l);
  CUDA_CHECK(cudaMemsetAsync(Zorth, 0, sizeof(math_t) * n * l, stream));

  // power sampling scheme
  for (int j = 1; j < q; j++) {
    if ((2 * j - 2) % s == 0) {
      qrGetQ(Y, Yorth, m, l, cusolverH, mgr);
      gemm(M, m, n, Yorth, Z, n, l, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta,
           cublasH);
    } else {
      gemm(M, m, n, Y, Z, n, l, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, cublasH);
    }

    if ((2 * j - 1) % s == 0) {
      qrGetQ(Z, Zorth, n, l, cusolverH, mgr);
      gemm(M, m, n, Zorth, Y, m, l, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta,
           cublasH);
    } else {
      gemm(M, m, n, Z, Y, m, l, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, cublasH);
    }
  }

  // orthogonalize on exit from loop to get Q
  math_t *Q = (math_t *)mgr.alloc(sizeof(math_t) * m * l);
  CUDA_CHECK(cudaMemsetAsync(Q, 0, sizeof(math_t) * m * l, stream));
  qrGetQ(Y, Q, m, l, cusolverH, mgr);

  // either QR of B^T method, or eigendecompose BB^T method
  if (!use_bbt) {
    // form Bt = Mt*Q : nxm * mxl = nxl
    math_t *Bt = (math_t *)mgr.alloc(sizeof(math_t) * n * l);
    CUDA_CHECK(cudaMemsetAsync(Bt, 0, sizeof(math_t) * n * l, stream));
    gemm(M, m, n, Q, Bt, n, l, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, cublasH);

    // compute QR factorization of Bt
    // M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */
    math_t *Qhat = (math_t *)mgr.alloc(sizeof(math_t) * n * l);
    CUDA_CHECK(cudaMemsetAsync(Qhat, 0, sizeof(math_t) * n * l, stream));
    math_t *Rhat = (math_t *)mgr.alloc(sizeof(math_t) * l * l);
    CUDA_CHECK(cudaMemsetAsync(Rhat, 0, sizeof(math_t) * l * l, stream));
    qrGetQR(Bt, Qhat, Rhat, n, l, cusolverH, mgr);

    // compute SVD of Rhat (lxl)
    math_t *Uhat = (math_t *)mgr.alloc(sizeof(math_t) * l * l);
    CUDA_CHECK(cudaMemsetAsync(Uhat, 0, sizeof(math_t) * l * l, stream));
    math_t *Vhat = (math_t *)mgr.alloc(sizeof(math_t) * l * l);
    CUDA_CHECK(cudaMemsetAsync(Vhat, 0, sizeof(math_t) * l * l, stream));
    if (use_jacobi)
      svdJacobi(Rhat, l, l, S_vec_tmp, Uhat, Vhat, true, true, tol, max_sweeps,
                cusolverH, mgr);
    else
      svdQR(Rhat, l, l, S_vec_tmp, Uhat, Vhat, true, true, cusolverH, cublasH,
            mgr);
    Matrix::sliceMatrix(S_vec_tmp, 1, l, S_vec, 0, 0, 1,
                        k); // First k elements of S_vec

    // Merge step 14 & 15 by calculating U = Q*Vhat[:,1:k] mxl * lxk = mxk
    if (gen_left_vec) {
      gemm(Q, m, l, Vhat, U, m, k /*used to be l and needs slicing*/,
           CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, cublasH);
    }

    // Merge step 14 & 15 by calculating V = Qhat*Uhat[:,1:k] nxl * lxk = nxk
    if (gen_right_vec) {
      gemm(Qhat, n, l, Uhat, V, n, k /*used to be l and needs slicing*/,
           CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, cublasH);
    }

    // clean up
    mgr.free(Rhat, stream);
    mgr.free(Qhat, stream);
    mgr.free(Uhat, stream);
    mgr.free(Vhat, stream);
    mgr.free(Bt, stream);

  } else {
    // build the matrix B B^T = Q^T M M^T Q column by column
    // Bt = M^T Q ; nxm * mxk = nxk
    math_t *B = (math_t *)mgr.alloc(sizeof(math_t) * n * l);
    gemm(Q, m, l, M, B, l, n, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, cublasH);

    math_t *BBt = (math_t *)mgr.alloc(sizeof(math_t) * l * l);
    gemm(B, l, n, B, BBt, l, l, CUBLAS_OP_N, CUBLAS_OP_T, alpha, beta, cublasH);

    // compute eigendecomposition of BBt
    math_t *Uhat = (math_t *)mgr.alloc(sizeof(math_t) * l * l);
    CUDA_CHECK(cudaMemsetAsync(Uhat, 0, sizeof(math_t) * l * l, stream));
    math_t *Uhat_dup = (math_t *)mgr.alloc(sizeof(math_t) * l * l);
    CUDA_CHECK(cudaMemsetAsync(Uhat_dup, 0, sizeof(math_t) * l * l, stream));
    Matrix::copyUpperTriangular(BBt, Uhat_dup, l, l);
    if (use_jacobi)
      eigJacobi(Uhat_dup, l, l, Uhat, S_vec_tmp, tol, max_sweeps, cusolverH,
                mgr);
    else
      eigDC(Uhat_dup, l, l, Uhat, S_vec_tmp, cusolverH, mgr);
    Matrix::seqRoot(S_vec_tmp, l);
    Matrix::sliceMatrix(S_vec_tmp, 1, l, S_vec, 0, p, 1,
                        l); // Last k elements of S_vec
    Matrix::colReverse(S_vec, 1, k);

    // Merge step 14 & 15 by calculating U = Q*Uhat[:,(p+1):l] mxl * lxk = mxk
    if (gen_left_vec) {
      gemm(Q, m, l, Uhat + p * l, U, m, k, CUBLAS_OP_N, CUBLAS_OP_N, alpha,
           beta, cublasH);
      Matrix::colReverse(U, m, k);
    }

    // Merge step 14 & 15 by calculating V = B^T Uhat[:,(p+1):l] *
    // Sigma^{-1}[(p+1):l, (p+1):l] nxl * lxk * kxk = nxk
    if (gen_right_vec) {
      math_t *Sinv = (math_t *)mgr.alloc(sizeof(math_t) * k * k);
      CUDA_CHECK(cudaMemsetAsync(Sinv, 0, sizeof(math_t) * k * k, stream));
      math_t *UhatSinv = (math_t *)mgr.alloc(sizeof(math_t) * l * k);
      CUDA_CHECK(cudaMemsetAsync(UhatSinv, 0, sizeof(math_t) * l * k, stream));
      Matrix::reciprocal(S_vec_tmp, l);
      Matrix::initializeDiagonalMatrix(S_vec_tmp + p, Sinv, k, k);

      gemm(Uhat + p * l, l, k, Sinv, UhatSinv, l, k, CUBLAS_OP_N, CUBLAS_OP_N,
           alpha, beta, cublasH);
      gemm(B, l, n, UhatSinv, V, n, k, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, cublasH);
      Matrix::colReverse(V, n, k);

      mgr.free(Sinv, stream);
      mgr.free(UhatSinv, stream);
    }

    // clean up
    mgr.free(BBt, stream);
    mgr.free(B, stream);
  }

  mgr.free(S_vec_tmp, stream);
  mgr.free(RN, stream);
  mgr.free(Y, stream);
  mgr.free(Q, stream);
  mgr.free(Z, stream);
  mgr.free(Yorth, stream);
  mgr.free(Zorth, stream);
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
 * @param mgr device allocator for temporary buffers during computation
 */
template <typename math_t>
void rsvdPerc(math_t *M, int n_rows, int n_cols, math_t *&S_vec, math_t *&U,
              math_t *&V, math_t PC_perc, math_t UpS_perc, bool use_bbt,
              bool gen_left_vec, bool gen_right_vec, bool use_jacobi,
              math_t tol, int max_sweeps, cusolverDnHandle_t cusolverH,
              cublasHandle_t cublasH, DeviceAllocator &mgr) {
  int k = max((int)(min(n_rows, n_cols) * PC_perc),
              1); // Number of singular values to be computed
  int p = max((int)(min(n_rows, n_cols) * UpS_perc), 1); // Upsamples
  rsvdFixedRank(M, n_rows, n_cols, S_vec, U, V, k, p, use_bbt, gen_left_vec,
                gen_right_vec, use_jacobi, tol, max_sweeps, cusolverH, cublasH,
                mgr);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
