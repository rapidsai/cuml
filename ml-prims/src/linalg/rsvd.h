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

#include "cusolver_wrappers.h"
#include "cublas_wrappers.h"
#include "cuda_utils.h"
#include "gemm.h"
#include "qr.h"
#include "eig.h"
#include "svd.h"
#include "transpose.h"
#include "../matrix/matrix.h"
#include "../matrix/math.h"
#include "../random/rng.h"

namespace MLCommon {
namespace LinAlg {


/**
 * @defgroup randomized singular value decomposition (RSVD) on the column major float type input matrix (Jacobi-based), by specifying no. of PCs and upsamples directly
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
 * @{
 */
template <typename math_t>
void rsvdFixedRank(	math_t *M, int n_rows, int n_cols,
			math_t* &S_vec, math_t* &U, math_t* &V,
			int k, int p, bool use_bbt,
			bool gen_left_vec, bool gen_right_vec, bool use_jacobi,
			math_t tol, int max_sweeps, cusolverDnHandle_t cusolverH, cublasHandle_t cublasH){

	// All the notations are following Algorithm 4 & 5 in S. Voronin's paper: https://arxiv.org/abs/1502.05366
	math_t *RN, *Y, *Z, *Q, *Yorth, *Zorth;
	math_t *Bt, *Qhat, *Rhat, *Uhat, *Vhat;
	math_t *S_vec_tmp;

	int m = n_rows, n = n_cols;
	int l = k + p; // Total number of singular values to be computed before truncation
	int q = 2; // Number of power sampling counts
	int s = 1; // Frequency controller for QR decomposition during power sampling scheme. s = 1: 2 QR per iteration; s = 2: 1 QR per iteration; s > 2: less frequent QR

	const math_t alpha = 1.0, beta = 0.0;

	// Build temporary U, S, V matrices
	allocate<math_t>(S_vec_tmp, l, true);

	// build random matrix
	allocate<math_t>(RN, n * l, true);
	Random::Rng<math_t> rng(484);
	rng.normal(RN, n * l, 0.0, alpha);

	// multiply to get matrix of random samples Y
	allocate<math_t>(Y, m * l, true);
	gemm(M, m, n, RN, Y, m, l, false, false, alpha, beta, cublasH);

	// now build up (M M^T)^q R
	allocate<math_t>(Z, n * l, true);
	allocate<math_t>(Yorth, m * l, true);
	allocate<math_t>(Zorth, n * l, true);

	// power sampling scheme
	for(int j = 1; j < q; j++){
		if((2*j - 2) % s == 0){
			qrGetQ(Y, Yorth, m, l, cusolverH);
			gemm(M, m, n, Yorth, Z, n, l, true, false, alpha, beta, cublasH);
		}
		else{
			gemm(M, m, n, Y, Z, n, l, true, false, alpha, beta, cublasH);
		}

		if((2*j - 1) % s == 0){
			qrGetQ(Z, Zorth, n, l, cusolverH);
			gemm(M, m, n, Zorth, Y, m, l, false, false, alpha, beta, cublasH);
		}
		else{
			gemm(M, m, n, Z, Y, m, l, false, false, alpha, beta, cublasH);
		}
	}

	// orthogonalize on exit from loop to get Q
	allocate<math_t>(Q, m * l, true);
	qrGetQ(Y, Q, m, l, cusolverH);

	// either QR of B^T method, or eigendecompose BB^T method
	if(!use_bbt){
		
		// form Bt = Mt*Q : nxm * mxl = nxl
		allocate<math_t>(Bt, n * l, true);
		gemm(M, m, n, Q, Bt, n, l, true, false, alpha, beta, cublasH);

		// compute QR factorization of Bt	
		//M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */
		allocate<math_t>(Qhat, n * l, true);
		allocate<math_t>(Rhat, l * l, true);
		qrGetQR(Bt, Qhat, Rhat, n, l, cusolverH);

		// compute SVD of Rhat (lxl)
		allocate<math_t>(Uhat, l * l, true);
		allocate<math_t>(Vhat, l * l, true);
		if (use_jacobi)
			svdJacobi(Rhat, l, l, S_vec_tmp, Uhat, Vhat, true, true, tol, max_sweeps, cusolverH);
		else
			svdQR(Rhat, l, l, S_vec_tmp, Uhat, Vhat, true, true, cusolverH);
		Matrix::sliceMatrix(S_vec_tmp, 1, l, S_vec, 0, 0, 1, k); // First k elements of S_vec

		// Merge step 14 & 15 by calculating U = Q*Vhat[:,1:k] mxl * lxk = mxk
		if (gen_left_vec) {
			gemm(Q, m, l, Vhat, U, m, k /*used to be l and needs slicing*/, false, false, alpha, beta, cublasH);
		}

		// Merge step 14 & 15 by calculating V = Qhat*Uhat[:,1:k] nxl * lxk = nxk
		if (gen_right_vec) {
			gemm(Qhat, n, l, Uhat, V, n, k /*used to be l and needs slicing*/, false, false, alpha, beta, cublasH);
		}

		// clean up
		CUDA_CHECK(cudaFree(Rhat));
		CUDA_CHECK(cudaFree(Qhat));
		CUDA_CHECK(cudaFree(Uhat));
		CUDA_CHECK(cudaFree(Vhat));
		CUDA_CHECK(cudaFree(Bt));

	} else {
		// build the matrix B B^T = Q^T M M^T Q column by column 
		// Bt = M^T Q ; nxm * mxk = nxk
		math_t *B;
		allocate<math_t>(B, n * l, true);
		gemm(Q, m, l, M, B, l, n, true, false, alpha, beta, cublasH);

		math_t *BBt;
		allocate<math_t>(BBt, l * l, true);
		gemm(B, l, n, B, BBt, l, l, false, true, alpha, beta, cublasH);

		// compute eigendecomposition of BBt
		math_t *Uhat, *Uhat_dup;
		allocate<math_t>(Uhat, l * l, true);
		allocate<math_t>(Uhat_dup, l * l, true);
		Matrix::copyUpperTriangular(BBt, Uhat_dup, l, l);
		if (use_jacobi)
			eigJacobi(Uhat_dup, l, l, Uhat, S_vec_tmp, tol, max_sweeps, cusolverH);
		else
			eigDC(Uhat_dup, l, l, Uhat, S_vec_tmp, cusolverH);
		Matrix::seqRoot(S_vec_tmp, l);
		Matrix::sliceMatrix(S_vec_tmp, 1, l, S_vec, 0, p, 1, l); // Last k elements of S_vec
		Matrix::colReverse(S_vec, 1, k);
		
		// Merge step 14 & 15 by calculating U = Q*Uhat[:,(p+1):l] mxl * lxk = mxk
		if (gen_left_vec) {
			gemm(Q, m, l, Uhat+p*l, U, m, k, false, false, alpha, beta, cublasH);
			Matrix::colReverse(U, m, k);
		}

		// Merge step 14 & 15 by calculating V = B^T Uhat[:,(p+1):l] * Sigma^{-1}[(p+1):l, (p+1):l] nxl * lxk * kxk = nxk
		if (gen_right_vec) {
			math_t *Sinv, *UhatSinv;
			allocate<math_t>(Sinv, k * k, true);
			allocate<math_t>(UhatSinv, l * k, true);
			Matrix::reciprocal(S_vec_tmp, l);
			Matrix::initializeDiagonalMatrix(S_vec_tmp+p, Sinv, k, k);

			gemm(Uhat+p*l, l, k, Sinv, UhatSinv, l, k, false, false, alpha, beta, cublasH);
			gemm(B, l, n, UhatSinv, V, n, k, true, false, alpha, beta, cublasH);
			Matrix::colReverse(V, n, k);

			CUDA_CHECK(cudaFree(Sinv));
			CUDA_CHECK(cudaFree(UhatSinv));
		}

		// clean up
		CUDA_CHECK(cudaFree(BBt));
	}

	CUDA_CHECK(cudaFree(S_vec_tmp));

	CUDA_CHECK(cudaFree(RN));
	CUDA_CHECK(cudaFree(Y));
	CUDA_CHECK(cudaFree(Q));
	CUDA_CHECK(cudaFree(Z));
	CUDA_CHECK(cudaFree(Yorth));
	CUDA_CHECK(cudaFree(Zorth));
}

/**
 * @defgroup randomized singular value decomposition (RSVD) on the column major float type input matrix (Jacobi-based), by specifying the PC and upsampling ratio
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
 * @{
 */
template <typename math_t>
void rsvdPerc(	math_t *M, int n_rows, int n_cols,
				math_t* &S_vec, math_t* &U, math_t* &V,
				math_t PC_perc, math_t UpS_perc, bool use_bbt,
				bool gen_left_vec, bool gen_right_vec, bool use_jacobi,
				math_t tol, int max_sweeps, cusolverDnHandle_t cusolverH, cublasHandle_t cublasH){
	int k = max((int) (min(n_rows, n_cols) * PC_perc), 1); // Number of singular values to be computed
    int p = max((int) (min(n_rows, n_cols) * UpS_perc), 1); // Upsamples
    rsvdFixedRank(M, n_rows, n_cols, S_vec, U, V, k, p, use_bbt, gen_left_vec, gen_right_vec, use_jacobi, tol, max_sweeps, cusolverH, cublasH);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
