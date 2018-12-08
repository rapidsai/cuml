#pragma once

#include "cusolver_wrappers.h"
#include "cublas_wrappers.h"
#include "cuda_utils.h"
#include "gemm.h"
#include "gemv.h"
#include "qr.h"
#include "eig.h"
#include "svd.h"
#include "transpose.h"
#include "matrix/matrix.h"
#include "matrix/math.h"
#include "random/rng.h"

namespace MLCommon {
namespace LinAlg {

template<typename math_t>
void lstsqSVD(math_t *A, int n_rows, int n_cols, math_t *b, math_t *w,
		cusolverDnHandle_t cusolverH, cublasHandle_t cublasH) {

	math_t *S, *V, *U;
	math_t *UT_b;

	int U_len = n_rows * n_rows;
	int V_len = n_cols * n_cols;

	allocate(U, U_len);
	allocate(V, V_len);
	allocate(S, n_cols);
	allocate(UT_b, n_rows);

	svdQR(A, n_rows, n_cols, S, U, V, true, true, cusolverH);

	gemv(U, n_rows, n_rows, b, UT_b, true, cublasH);

	Matrix::truncZeroOrigin(UT_b, n_rows, w, n_cols, 1);
	Matrix::matrixVectorBinaryDivSkipZero(w, S, 1, n_cols, false, true);

	gemv(V, n_cols, n_cols, w, w, false, cublasH);

	CUDA_CHECK(cudaFree(U));
	CUDA_CHECK(cudaFree(V));
	CUDA_CHECK(cudaFree(S));
	CUDA_CHECK(cudaFree(UT_b));
}

template<typename math_t>
void lstsqEig(math_t *A, int n_rows, int n_cols, math_t *b, math_t *w,
		cusolverDnHandle_t cusolverH, cublasHandle_t cublasH) {

	math_t *S, *V, *U;

	int U_len = n_rows * n_cols;
	int V_len = n_cols * n_cols;

	allocate(U, U_len);
	allocate(V, V_len);
	allocate(S, n_cols);

	svdEig(A, n_rows, n_cols, S, U, V, true, cublasH, cusolverH);

	gemv(U, n_rows, n_cols, b, w, true, cublasH);

	Matrix::matrixVectorBinaryDivSkipZero(w, S, 1, n_cols, false, true);

	gemv(V, n_cols, n_cols, w, w, false, cublasH);

	CUDA_CHECK(cudaFree(U));
	CUDA_CHECK(cudaFree(V));
	CUDA_CHECK(cudaFree(S));
}



template<typename math_t>
void lstsqQR(math_t *A, int n_rows, int n_cols, math_t *b, math_t *w,
		cusolverDnHandle_t cusolverH, cublasHandle_t cublasH) {

	int m = n_rows;
	int n = n_cols;

	math_t *d_tau = NULL;
	int *d_info = NULL;
	int info = 0;
	math_t *d_work = NULL;
	CUDA_CHECK(cudaMalloc((void **)&d_info, sizeof(int)));
	CUDA_CHECK(cudaMalloc((void **)&d_tau, sizeof(math_t)*n));

	const cublasSideMode_t side = CUBLAS_SIDE_LEFT;
	const cublasOperation_t trans = CUBLAS_OP_T;

	int lwork_geqrf = 0;
	int lwork_ormqr = 0;
	int lwork = 0;

	const int lda = m;
	const int ldb = m;

	CUSOLVER_CHECK(
			cusolverDngeqrf_bufferSize(cusolverH, m, n, A, lda, &lwork_geqrf));

	CUSOLVER_CHECK(cusolverDnormqr_bufferSize(cusolverH, side,
	                                          trans,
	                                          m,
	                                          1,
	                                          n,
	                                          A, lda, d_tau, b, // C,
			                                  lda, // ldc,
			                                  &lwork_ormqr));

	lwork = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;

	CUDA_CHECK(cudaMalloc(&d_work, sizeof(math_t) * lwork));

	CUSOLVER_CHECK(
			cusolverDngeqrf(cusolverH, m, n, A, lda, d_tau, d_work, lwork,
					d_info));

	CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
	ASSERT(0 == info, "lstsq.h: QR wasn't successful");

	CUSOLVER_CHECK(cusolverDnormqr(
			cusolverH,
	        side,
	        trans,
	        m,
	        1,
	        n,
	        A,
	        lda,
	        d_tau,
	        b,
	        ldb,
	        d_work,
	        lwork,
	        d_info));

	CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT(0 == info, "lstsq.h: QR wasn't successful");

    const math_t one = 1;

    CUBLAS_CHECK(cublastrsm(
             cublasH,
             side,
             CUBLAS_FILL_MODE_UPPER,
             CUBLAS_OP_N,
             CUBLAS_DIAG_NON_UNIT,
             n,
             1,
             &one,
             A,
             lda,
             b,
             ldb));

    CUDA_CHECK(cudaMemcpy(w, b, sizeof(math_t) * n, cudaMemcpyDeviceToDevice));

    if (NULL != d_tau)   cudaFree(d_tau);
    if (NULL != d_info)  cudaFree(d_info);
    if (NULL != d_work)  cudaFree(d_work);
}


}
;
// end namespace LinAlg
}
;
// end namespace MLCommon
