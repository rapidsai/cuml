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

#include "ml_utils.h"
#include <linalg/svd.h>
#include <linalg/gemm.h>
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <stats/stddev.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/norm.h>
#include <stats/sum.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include "preprocess.h"

namespace ML {
namespace GLM {

using namespace MLCommon;

template<typename math_t>
void ridgeSolve(math_t *S, math_t *V, math_t *U, int n_rows, int n_cols,
		math_t *b, math_t *alpha, int n_alpha, math_t *w,
		cusolverDnHandle_t cusolverH, cublasHandle_t cublasH) {

	// Implements this: w = V * inv(S^2 + λ*I) * S * U^T * b
	math_t *r;
	math_t alp = math_t(1);
	math_t beta = math_t(0);

	allocate(r, n_cols, true);
	copy(r, S, n_cols);
	Matrix::power(r, n_cols);

	LinAlg::addScalar(r, r, alpha[0], n_cols);

	Matrix::reciprocal(r, n_cols);
	Matrix::matrixVectorBinaryMult(V, r, n_cols, n_cols, false);
	Matrix::matrixVectorBinaryMult(V, S, n_cols, n_cols, false);

	LinAlg::gemm(V, n_cols, n_cols, U, U, n_cols, n_rows, false, true, alp,
			beta, cublasH);

	LinAlg::gemm(U, n_cols, n_rows, b, w, n_cols, 1, false, false, alp, beta,
			cublasH);

	CUDA_CHECK(cudaFree(r));
}

template<typename math_t>
void ridgeSVD(math_t *A, int n_rows, int n_cols, math_t *b, math_t *alpha,
		int n_alpha, math_t *w, cusolverDnHandle_t cusolverH,
		cublasHandle_t cublasH) {

	math_t *S, *V, *U;

	int U_len = n_rows * n_rows;
	int V_len = n_cols * n_cols;

	allocate(U, U_len);
	allocate(V, V_len);
	allocate(S, n_cols);

	LinAlg::svdQR(A, n_rows, n_cols, S, U, V, true, true, cusolverH);
	ridgeSolve(S, V, U, n_rows, n_cols, b, alpha, n_alpha, w, cusolverH,
			cublasH);

	CUDA_CHECK(cudaFree(U));
	CUDA_CHECK(cudaFree(V));
	CUDA_CHECK(cudaFree(S));

}

template<typename math_t>
void ridgeEig(math_t *A, int n_rows, int n_cols, math_t *b, math_t *alpha,
		int n_alpha, math_t *w, cusolverDnHandle_t cusolverH,
		cublasHandle_t cublasH) {

	math_t *S, *V, *U;

	int U_len = n_rows * n_cols;
	int V_len = n_cols * n_cols;

	allocate(U, U_len);
	allocate(V, V_len);
	allocate(S, n_cols);

	LinAlg::svdEig(A, n_rows, n_cols, S, U, V, true, cublasH, cusolverH);
	ridgeSolve(S, V, U, n_rows, n_cols, b, alpha, n_alpha, w, cusolverH,
			cublasH);

	CUDA_CHECK(cudaFree(U));
	CUDA_CHECK(cudaFree(V));
	CUDA_CHECK(cudaFree(S));
}

template<typename math_t>
void ridgeFit(math_t *input, int n_rows, int n_cols, math_t *labels,
		math_t *alpha, int n_alpha, math_t *coef, math_t *intercept,
		bool fit_intercept, bool normalize, cublasHandle_t cublas_handle,
		cusolverDnHandle_t cusolver_handle, int algo = 0) {

	ASSERT(n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than two");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

	math_t *mu_input, *norm2_input, *mu_labels;

	if (fit_intercept) {
		allocate(mu_input, n_cols);
		allocate(mu_labels, 1);
		if (normalize) {
			allocate(norm2_input, n_cols);
		}
		preProcessData(input, n_rows, n_cols, labels, intercept, mu_input,
				mu_labels, norm2_input, fit_intercept, normalize, cublas_handle,
				cusolver_handle);
	}

	if (algo == 0) {
		ridgeSVD(input, n_rows, n_cols, labels, alpha, n_alpha, coef,
				cusolver_handle, cublas_handle);
	} else if (algo == 1) {
		ridgeEig(input, n_rows, n_cols, labels, alpha, n_alpha, coef,
				cusolver_handle, cublas_handle);
	} else if (algo == 2) {
		ASSERT(false,
				"ridgeFit: no algorithm with this id has been implemented");
	} else {
		ASSERT(false,
				"ridgeFit: no algorithm with this id has been implemented");
	}

	if (fit_intercept) {
		postProcessData(input, n_rows, n_cols, labels, coef, intercept,
				mu_input, mu_labels, norm2_input, fit_intercept, normalize,
				cublas_handle, cusolver_handle);

		if (normalize) {
			if (norm2_input != NULL)
				cudaFree(norm2_input);
		}

		if (mu_input != NULL)
			cudaFree(mu_input);
		if (mu_labels != NULL)
			cudaFree(mu_labels);
	} else {
		*intercept = math_t(0);
	}

}

template<typename math_t>
void ridgePredict(const math_t *input, int n_rows, int n_cols,
		const math_t *coef, math_t intercept, math_t *preds,
		cublasHandle_t cublas_handle) {

	ASSERT(n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than two");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

	math_t alpha = math_t(1);
	math_t beta = math_t(0);
	LinAlg::gemm(input, n_rows, n_cols, coef, preds, n_rows, 1, false, false,
			alpha, beta, cublas_handle);

	LinAlg::addScalar(preds, preds, intercept, n_rows * n_cols);

}

/** @} */
}
;
}
;
// end namespace ML
