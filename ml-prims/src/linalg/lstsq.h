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
#include "gemv.h"
#include "qr.h"
#include "eig.h"
#include "svd.h"
#include "transpose.h"
#include "matrix/matrix.h"
#include "matrix/math.h"
#include "random/rng.h"
#include "common/device_buffer.hpp"

namespace MLCommon {
namespace LinAlg {

template<typename math_t>
void lstsqSVD(math_t *A, int n_rows, int n_cols, math_t *b, math_t *w,
              cusolverDnHandle_t cusolverH, cublasHandle_t cublasH,
              std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream) {

	ASSERT(n_cols > 0,
			"lstsq: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"lstsq: number of rows cannot be less than two");

	int U_len = n_rows * n_cols;
	int V_len = n_cols * n_cols;

        device_buffer<math_t> S(allocator, stream, n_cols);
        device_buffer<math_t> V(allocator, stream, V_len);
        device_buffer<math_t> U(allocator, stream, U_len);
        device_buffer<math_t> UT_b(allocator, stream, n_rows);

	svdQR(A, n_rows, n_cols, S.data(), U.data(), V.data(), true, true, true,
              cusolverH, cublasH, allocator, stream);

	gemv(U.data(), n_rows, n_cols, b, w, true, cublasH, stream);

	Matrix::matrixVectorBinaryDivSkipZero(w, S.data(), 1, n_cols, false, true, stream);

	gemv(V.data(), n_cols, n_cols, w, w, false, cublasH, stream);
}

template<typename math_t>
void lstsqEig(math_t *A, int n_rows, int n_cols, math_t *b, math_t *w,
              cusolverDnHandle_t cusolverH, cublasHandle_t cublasH,
              std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream) {

	ASSERT(n_cols > 1,
			"lstsq: number of columns cannot be less than two");
	ASSERT(n_rows > 1,
			"lstsq: number of rows cannot be less than two");

	int U_len = n_rows * n_cols;
	int V_len = n_cols * n_cols;

        device_buffer<math_t> S(allocator, stream, n_cols);
        device_buffer<math_t> V(allocator, stream, V_len);
        device_buffer<math_t> U(allocator, stream, U_len);

	svdEig(A, n_rows, n_cols, S.data(), U.data(), V.data(), true, cublasH,
               cusolverH, stream, allocator);

	gemv(U.data(), n_rows, n_cols, b, w, true, cublasH, stream);

	Matrix::matrixVectorBinaryDivSkipZero(w, S.data(), 1, n_cols, false, true, stream);

	gemv(V.data(), n_cols, n_cols, w, w, false, cublasH, stream);
}


template<typename math_t>
void lstsqQR(math_t *A, int n_rows, int n_cols, math_t *b, math_t *w,
		cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, cudaStream_t stream) {

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
					d_info, stream));

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
	        d_info,
          stream));

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
             ldb,
             stream));

    CUDA_CHECK(cudaMemcpyAsync(w, b, sizeof(math_t) * n, cudaMemcpyDeviceToDevice, stream));

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
