/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <raft/cuda_utils.cuh>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/qr.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/linalg/transpose.h>
#include <raft/matrix/math.hpp>
#include <raft/matrix/matrix.hpp>
#include <raft/mr/device/buffer.hpp>
#include <raft/random/rng.hpp>

namespace MLCommon {
namespace LinAlg {

/**
 * @brief randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying no. of PCs and
 * upsamples directly
 * @param handle: raft handle
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param k: no. of singular values to be computed
 * @param p: no. of upsamples
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param gen_left_vec: left vector needs to be generated or not?
 * @param gen_right_vec: right vector needs to be generated or not?
 * @param use_jacobi: whether to jacobi solver for decomposition
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param stream cuda stream
 */
template <typename math_t>
void rsvdFixedRank(const raft::handle_t& handle,
                   math_t* M,
                   int n_rows,
                   int n_cols,
                   math_t* S_vec,
                   math_t* U,
                   math_t* V,
                   int k,
                   int p,
                   bool use_bbt,
                   bool gen_left_vec,
                   bool gen_right_vec,
                   bool use_jacobi,
                   math_t tol,
                   int max_sweeps,
                   cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();
  cublasHandle_t cublasH       = handle.get_cublas_handle();

  // All the notations are following Algorithm 4 & 5 in S. Voronin's paper:
  // https://arxiv.org/abs/1502.05366

  int m = n_rows, n = n_cols;
  int l = k + p;  // Total number of singular values to be computed before truncation
  int q = 2;      // Number of power sampling counts
  int s = 1;      // Frequency controller for QR decomposition during power sampling
                  // scheme. s = 1: 2 QR per iteration; s = 2: 1 QR per iteration; s
                  // > 2: less frequent QR

  const math_t alpha = 1.0, beta = 0.0;

  // Build temporary U, S, V matrices
  rmm::device_uvector<math_t> S_vec_tmp(l, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(S_vec_tmp.data(), 0, sizeof(math_t) * l, stream));

  // build random matrix
  rmm::device_uvector<math_t> RN(n * l, stream);
  raft::random::Rng rng(484);
  rng.normal(RN.data(), n * l, math_t(0.0), alpha, stream);

  // multiply to get matrix of random samples Y
  rmm::device_uvector<math_t> Y(m * l, stream);
  raft::linalg::gemm(
    handle, M, m, n, RN.data(), Y.data(), m, l, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, stream);

  // now build up (M M^T)^q R
  rmm::device_uvector<math_t> Z(n * l, stream);
  rmm::device_uvector<math_t> Yorth(m * l, stream);
  rmm::device_uvector<math_t> Zorth(n * l, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(Z.data(), 0, sizeof(math_t) * n * l, stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(Yorth.data(), 0, sizeof(math_t) * m * l, stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(Zorth.data(), 0, sizeof(math_t) * n * l, stream));

  // power sampling scheme
  for (int j = 1; j < q; j++) {
    if ((2 * j - 2) % s == 0) {
      raft::linalg::qrGetQ(handle, Y.data(), Yorth.data(), m, l, stream);
      raft::linalg::gemm(handle,
                         M,
                         m,
                         n,
                         Yorth.data(),
                         Z.data(),
                         n,
                         l,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         alpha,
                         beta,
                         stream);
    } else {
      raft::linalg::gemm(
        handle, M, m, n, Y.data(), Z.data(), n, l, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, stream);
    }

    if ((2 * j - 1) % s == 0) {
      raft::linalg::qrGetQ(handle, Z.data(), Zorth.data(), n, l, stream);
      raft::linalg::gemm(handle,
                         M,
                         m,
                         n,
                         Zorth.data(),
                         Y.data(),
                         m,
                         l,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         alpha,
                         beta,
                         stream);
    } else {
      raft::linalg::gemm(
        handle, M, m, n, Z.data(), Y.data(), m, l, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, stream);
    }
  }

  // orthogonalize on exit from loop to get Q
  rmm::device_uvector<math_t> Q(m * l, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(Q.data(), 0, sizeof(math_t) * m * l, stream));
  raft::linalg::qrGetQ(handle, Y.data(), Q.data(), m, l, stream);

  // either QR of B^T method, or eigendecompose BB^T method
  if (!use_bbt) {
    // form Bt = Mt*Q : nxm * mxl = nxl
    rmm::device_uvector<math_t> Bt(n * l, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(Bt.data(), 0, sizeof(math_t) * n * l, stream));
    raft::linalg::gemm(
      handle, M, m, n, Q.data(), Bt.data(), n, l, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, stream);

    // compute QR factorization of Bt
    // M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */
    rmm::device_uvector<math_t> Qhat(n * l, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(Qhat.data(), 0, sizeof(math_t) * n * l, stream));
    rmm::device_uvector<math_t> Rhat(l * l, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(Rhat.data(), 0, sizeof(math_t) * l * l, stream));
    raft::linalg::qrGetQR(handle, Bt.data(), Qhat.data(), Rhat.data(), n, l, stream);

    // compute SVD of Rhat (lxl)
    rmm::device_uvector<math_t> Uhat(l * l, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(Uhat.data(), 0, sizeof(math_t) * l * l, stream));
    rmm::device_uvector<math_t> Vhat(l * l, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(Vhat.data(), 0, sizeof(math_t) * l * l, stream));
    if (use_jacobi)
      raft::linalg::svdJacobi(handle,
                              Rhat.data(),
                              l,
                              l,
                              S_vec_tmp.data(),
                              Uhat.data(),
                              Vhat.data(),
                              true,
                              true,
                              tol,
                              max_sweeps,
                              stream);
    else
      raft::linalg::svdQR(handle,
                          Rhat.data(),
                          l,
                          l,
                          S_vec_tmp.data(),
                          Uhat.data(),
                          Vhat.data(),
                          true,
                          true,
                          true,
                          stream);
    raft::matrix::sliceMatrix(S_vec_tmp.data(),
                              1,
                              l,
                              S_vec,
                              0,
                              0,
                              1,
                              k,
                              stream);  // First k elements of S_vec

    // Merge step 14 & 15 by calculating U = Q*Vhat[:,1:k] mxl * lxk = mxk
    if (gen_left_vec) {
      raft::linalg::gemm(handle,
                         Q.data(),
                         m,
                         l,
                         Vhat.data(),
                         U,
                         m,
                         k /*used to be l and needs slicing*/,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         alpha,
                         beta,
                         stream);
    }

    // Merge step 14 & 15 by calculating V = Qhat*Uhat[:,1:k] nxl * lxk = nxk
    if (gen_right_vec) {
      raft::linalg::gemm(handle,
                         Qhat.data(),
                         n,
                         l,
                         Uhat.data(),
                         V,
                         n,
                         k /*used to be l and needs slicing*/,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         alpha,
                         beta,
                         stream);
    }
  } else {
    // build the matrix B B^T = Q^T M M^T Q column by column
    // Bt = M^T Q ; nxm * mxk = nxk
    rmm::device_uvector<math_t> B(n * l, stream);
    raft::linalg::gemm(
      handle, Q.data(), m, l, M, B.data(), l, n, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta, stream);

    rmm::device_uvector<math_t> BBt(l * l, stream);
    raft::linalg::gemm(handle,
                       B.data(),
                       l,
                       n,
                       B.data(),
                       BBt.data(),
                       l,
                       l,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       alpha,
                       beta,
                       stream);

    // compute eigendecomposition of BBt
    rmm::device_uvector<math_t> Uhat(l * l, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(Uhat.data(), 0, sizeof(math_t) * l * l, stream));
    rmm::device_uvector<math_t> Uhat_dup(l * l, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(Uhat_dup.data(), 0, sizeof(math_t) * l * l, stream));
    raft::matrix::copyUpperTriangular(BBt.data(), Uhat_dup.data(), l, l, stream);
    if (use_jacobi)
      raft::linalg::eigJacobi(
        handle, Uhat_dup.data(), l, l, Uhat.data(), S_vec_tmp.data(), stream, tol, max_sweeps);
    else
      raft::linalg::eigDC(handle, Uhat_dup.data(), l, l, Uhat.data(), S_vec_tmp.data(), stream);
    raft::matrix::seqRoot(S_vec_tmp.data(), l, stream);
    raft::matrix::sliceMatrix(S_vec_tmp.data(),
                              1,
                              l,
                              S_vec,
                              0,
                              p,
                              1,
                              l,
                              stream);  // Last k elements of S_vec
    raft::matrix::colReverse(S_vec, 1, k, stream);

    // Merge step 14 & 15 by calculating U = Q*Uhat[:,(p+1):l] mxl * lxk = mxk
    if (gen_left_vec) {
      raft::linalg::gemm(handle,
                         Q.data(),
                         m,
                         l,
                         Uhat.data() + p * l,
                         U,
                         m,
                         k,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         alpha,
                         beta,
                         stream);
      raft::matrix::colReverse(U, m, k, stream);
    }

    // Merge step 14 & 15 by calculating V = B^T Uhat[:,(p+1):l] *
    // Sigma^{-1}[(p+1):l, (p+1):l] nxl * lxk * kxk = nxk
    if (gen_right_vec) {
      rmm::device_uvector<math_t> Sinv(k * k, stream);
      RAFT_CUDA_TRY(cudaMemsetAsync(Sinv.data(), 0, sizeof(math_t) * k * k, stream));
      rmm::device_uvector<math_t> UhatSinv(l * k, stream);
      RAFT_CUDA_TRY(cudaMemsetAsync(UhatSinv.data(), 0, sizeof(math_t) * l * k, stream));
      raft::matrix::reciprocal(S_vec_tmp.data(), l, stream);
      raft::matrix::initializeDiagonalMatrix(S_vec_tmp.data() + p, Sinv.data(), k, k, stream);

      raft::linalg::gemm(handle,
                         Uhat.data() + p * l,
                         l,
                         k,
                         Sinv.data(),
                         UhatSinv.data(),
                         l,
                         k,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         alpha,
                         beta,
                         stream);
      raft::linalg::gemm(handle,
                         B.data(),
                         l,
                         n,
                         UhatSinv.data(),
                         V,
                         n,
                         k,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         alpha,
                         beta,
                         stream);
      raft::matrix::colReverse(V, n, k, stream);
    }
  }
}

/**
 * @brief randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying the PC and upsampling
 * ratio
 * @param handle: raft handle
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param PC_perc: percentage of singular values to be computed
 * @param UpS_perc: upsampling percentage
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param gen_left_vec: left vector needs to be generated or not?
 * @param gen_right_vec: right vector needs to be generated or not?
 * @param use_jacobi: whether to jacobi solver for decomposition
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param stream cuda stream
 */
template <typename math_t>
void rsvdPerc(const raft::handle_t& handle,
              math_t* M,
              int n_rows,
              int n_cols,
              math_t* S_vec,
              math_t* U,
              math_t* V,
              math_t PC_perc,
              math_t UpS_perc,
              bool use_bbt,
              bool gen_left_vec,
              bool gen_right_vec,
              bool use_jacobi,
              math_t tol,
              int max_sweeps,
              cudaStream_t stream)
{
  int k = max((int)(min(n_rows, n_cols) * PC_perc),
              1);  // Number of singular values to be computed
  int p = max((int)(min(n_rows, n_cols) * UpS_perc), 1);  // Upsamples
  rsvdFixedRank(handle,
                M,
                n_rows,
                n_cols,
                S_vec,
                U,
                V,
                k,
                p,
                use_bbt,
                gen_left_vec,
                gen_right_vec,
                use_jacobi,
                tol,
                max_sweeps,
                stream);
}

};  // end namespace LinAlg
};  // end namespace MLCommon
