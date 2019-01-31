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

#include "cublas_wrappers.h"
#include "cusolver_wrappers.h"
#include "cuda_utils.h"
#include "gemm.h"
#include "eig.h"
#include "transpose.h"
#include "matrix/matrix.h"
#include "matrix/math.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup singular value decomposition (SVD) on the column major float type input matrix using QR method
 * @param in: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param sing_vals: singular values of input matrix
 * @param left_sing_vecs: left singular values of input matrix
 * @param right_sing_vecs: right singular values of input matrix
 * @param gen_left_vec: generate left eig vector. Not activated.
 * @param gen_right_vec: generate right eig vector. Not activated.
 * @{
 */

// TODO: activate gen_left_vec and gen_right_vec options
// TODO: couldn't template this function due to cusolverDnSgesvd and cusolverSnSgesvd. Check if there is any other way.

template <typename T>
void svdQR(T* in, int n_rows, int n_cols, T* sing_vals,
		   T* left_sing_vecs, T* right_sing_vecs,
		   bool gen_left_vec, bool gen_right_vec, cusolverDnHandle_t cusolverH) {

	const int m = n_rows;
	const int n = n_cols;

	int *devInfo = NULL;
	T *d_work = NULL;
	T *d_rwork = NULL;
	T *d_W = NULL;

	// The transposed right singular vector
	T *right_sing_vecs_trans;
	CUDA_CHECK(cudaMalloc((void** ) &right_sing_vecs_trans, sizeof(T) * n * n));

	int lwork = 0;
	CUDA_CHECK(cudaMalloc((void** ) &devInfo, sizeof(int)));

	CUSOLVER_CHECK(
			cusolverDngesvd_bufferSize<T>(cusolverH, n_rows,
					n_cols, &lwork));

	CUDA_CHECK(cudaMalloc((void** ) &d_work, sizeof(T) * lwork));

	signed char jobu = 'A';
	signed char jobvt = 'A';

	CUSOLVER_CHECK(
			cusolverDngesvd(cusolverH, jobu, jobvt, m,
					n, in, m, sing_vals,
					left_sing_vecs, m,
					right_sing_vecs_trans, n,
					d_work, lwork, d_rwork, devInfo));


	CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_rwork));
	CUDA_CHECK(cudaFree(d_W));

	// Transpose the right singular vector back
	cublasHandle_t cublasH;
	CUBLAS_CHECK(cublasCreate(&cublasH));
	transpose(right_sing_vecs_trans, right_sing_vecs, n, n, cublasH);
	CUBLAS_CHECK(cublasDestroy(cublasH));

	CUDA_CHECK(cudaGetLastError());

	int d_dev_info;
	updateHost(&d_dev_info, devInfo, 1);

	CUDA_CHECK(cudaFree(devInfo));
	CUDA_CHECK(cudaGetLastError());

	ASSERT(d_dev_info == 0,
			"svd.h: svd couldn't converge to a solution. "
					"This usually occurs when some of the features do not vary enough. "
					"Please try with more data that has variability");
}

template <typename T>
void svdEig(T* in, int n_rows, int n_cols, T* S,
		   T* U, T* V, bool gen_left_vec,
		   cublasHandle_t cublasH, cusolverDnHandle_t cusolverH) {

	T *in_cross_mult;
	int len = n_cols * n_cols;
	allocate(in_cross_mult, len);

	T alpha = T(1);
	T beta = T(0);
	gemm(in, n_rows, n_cols, in, in_cross_mult, n_cols, n_cols, true, false, alpha, beta, cublasH);

	eigDC(in_cross_mult, n_cols, n_cols, V, S, cusolverH);

	Matrix::colReverse(V, n_cols, n_cols);
	Matrix::rowReverse(S, n_cols, 1);

	Matrix::seqRoot(S, S, alpha, n_cols, true);

    if (gen_left_vec) {
    	gemm(in, n_rows, n_cols, V, U, n_rows, n_cols, false, false, alpha, beta, cublasH);
    	Matrix::matrixVectorBinaryDivSkipZero(U, S, n_rows, n_cols, false, true);
    }

	CUDA_CHECK(cudaFree(in_cross_mult));
}

/**
 * @defgroup singular value decomposition (SVD) on the column major input matrix using Jacobi method
 * @param in: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param sing_vals: singular values of input matrix
 * @param left_sing_vecs: left singular vectors of input matrix
 * @param right_sing_vecs_trans: right singular vectors of input matrix
 * @param gen_left_vec: generate left eig vector. Not activated.
 * @param gen_right_vec: generate right eig vector. Not activated.
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the error is below tol
 * @param sweeps: number of sweeps in the Jacobi algorithm. The more the better accuracy.
 * @{
 */

template <typename math_t>
void svdJacobi(math_t* in, int n_rows, int n_cols, math_t* sing_vals,
		   math_t* left_sing_vecs, math_t* right_sing_vecs,
		   bool gen_left_vec, bool gen_right_vec, math_t tol, int max_sweeps, cusolverDnHandle_t cusolverH) {

	cudaStream_t stream = NULL;
	gesvdjInfo_t gesvdj_params = NULL;

	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
	CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
	CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
	CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

	int m = n_rows;
	int n = n_cols;

	int *devInfo = NULL;
	math_t *d_work = NULL;
	math_t *d_rwork = NULL;
	math_t *d_W = NULL;

	int lwork = 0;
	int econ = 1;

	CUDA_CHECK(cudaMalloc((void** ) &devInfo, sizeof(int)));

	CUSOLVER_CHECK(
			cusolverDngesvdj_bufferSize(cusolverH,
					CUSOLVER_EIG_MODE_VECTOR, econ, m, n,
					in, m, sing_vals, left_sing_vecs, m, right_sing_vecs, n,
					&lwork, gesvdj_params));

	CUDA_CHECK(cudaMalloc((void** ) &d_work, sizeof(math_t) * lwork));

	CUSOLVER_CHECK(
			cusolverDngesvdj(cusolverH,
					CUSOLVER_EIG_MODE_VECTOR, econ, m, n,
					in, m, sing_vals,
					left_sing_vecs, m, right_sing_vecs, n, d_work,
					lwork, devInfo, gesvdj_params));

	if (devInfo)
		CUDA_CHECK(cudaFree(devInfo));
	if (d_work)
		CUDA_CHECK(cudaFree(d_work));
	if (d_rwork)
		CUDA_CHECK(cudaFree(d_rwork));
	if (d_W)
		CUDA_CHECK(cudaFree(d_W));

	CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

/**
 * @defgroup reconstruct a matrix use left and right singular vectors and singular values
 * @param U: left singular vectors of size n_rows x k
 * @param S: square matrix with singular values on its diagonal, k x k
 * @param VT: right singular vectors of size n_cols x k
 * @param out: reconstructed matrix to be returned
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values
 * @{
 */
template <typename math_t>
void svdReconstruction(math_t *U, math_t *S, math_t *V, math_t *out, int n_rows, int n_cols, int k){
	cublasHandle_t cublasH;
	CUBLAS_CHECK(cublasCreate(&cublasH));

	const math_t alpha = 1.0, beta = 0.0;
	math_t *SVT;
	allocate<math_t>(SVT, k * n_cols * sizeof(math_t), true);

	gemm(S, k, k, V, SVT, k, n_cols, false, true, alpha, beta, cublasH);
	gemm(U, n_rows, k, SVT, out, n_rows, n_cols, false, false, alpha, beta, cublasH);

	CUDA_CHECK(cudaFree(SVT));
	CUBLAS_CHECK(cublasDestroy(cublasH));
}

/**
 * @defgroup reconstruct a matrix use left and right singular vectors and singular values
 * @param U: left singular vectors of size n_rows x k
 * @param S_vec: singular values as a vector
 * @param VT: right singular vectors of size n_cols x k
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values to be computed, 1.0 for normal SVD
 * @param tol: tolerance for the evaluation
 * @{
 */
template <typename math_t>
bool evaluateSVDByL2Norm(math_t *A_d, math_t *U, math_t *S_vec, math_t *V, int n_rows, int n_cols, int k, math_t tol) {
	int m = n_rows, n = n_cols;

	// form product matrix
	math_t *P_d, *S_mat;
	allocate<math_t>(P_d, m * n * sizeof(math_t), true);
	allocate<math_t>(S_mat, k * k * sizeof(math_t), true);

	Matrix::initializeDiagonalMatrix(S_vec, S_mat, k, k);
	svdReconstruction(U, S_mat, V, P_d, m, n, k);

	// get norms of each
	math_t normA = Matrix::getL2Norm(A_d, m * n);
	math_t normU = Matrix::getL2Norm(U, m * k);
	math_t normS = Matrix::getL2Norm(S_mat, k * k);
	math_t normV = Matrix::getL2Norm(V, n * k);
	math_t normP = Matrix::getL2Norm(P_d, m * n);

	// calculate percent error
	const math_t alpha = 1.0, beta = -1.0;
	math_t *A_minus_P;
	allocate<math_t>(A_minus_P, m * n * sizeof(math_t), true);

	cublasHandle_t cublasH;
	CUBLAS_CHECK(cublasCreate(&cublasH));
	CUBLAS_CHECK(cublasgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
							m, n,
							&alpha, A_d, m,
							&beta, P_d, m,
							A_minus_P, m));
	CUBLAS_CHECK(cublasDestroy(cublasH));

	math_t norm_A_minus_P = Matrix::getL2Norm(A_minus_P, m * n);
	math_t percent_error = 100.0 * norm_A_minus_P / normA;

	CUDA_CHECK(cudaFree(A_minus_P));
	CUDA_CHECK(cudaFree(P_d));
	CUDA_CHECK(cudaFree(S_mat));

	return (percent_error / 100.0 < tol);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
