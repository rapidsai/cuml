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
#include "common/cumlHandle.hpp"

namespace ML {
namespace GLM {

using namespace MLCommon;

template<typename math_t>
void ridgeSolve(math_t *S, math_t *V, math_t *U, int n_rows, int n_cols,
		math_t *b, math_t *alpha, int n_alpha, math_t *w,
		cusolverDnHandle_t cusolverH, cublasHandle_t cublasH) {

	// Implements this: w = V * inv(S^2 + λ*I) * S * U^T * b
	math_t *S_nnz;
	math_t alp = math_t(1);
	math_t beta = math_t(0);
	math_t thres = math_t(1e-10);

	Matrix::setSmallValuesZero(S, n_cols, thres);
	allocate(S_nnz, n_cols, true);
	copy(S_nnz, S, n_cols);
	Matrix::power(S_nnz, n_cols);
	LinAlg::addScalar(S_nnz, S_nnz, alpha[0], n_cols);
	Matrix::matrixVectorBinaryDivSkipZero(S, S_nnz, 1, n_cols, false, true, true);

	Matrix::matrixVectorBinaryMult(V, S, n_cols, n_cols, false, true);
	LinAlg::gemm(U, n_rows, n_cols, b, S_nnz, n_cols, 1, CUBLAS_OP_T, CUBLAS_OP_N, alp, beta,
			cublasH);

	LinAlg::gemm(V, n_cols, n_cols, S_nnz, w, n_cols, 1, CUBLAS_OP_N, CUBLAS_OP_N, alp,
			beta, cublasH);

	CUDA_CHECK(cudaFree(S_nnz));
}

template<typename math_t>
void ridgeSVD(math_t *A, int n_rows, int n_cols, math_t *b, math_t *alpha,
		int n_alpha, math_t *w, const cumlHandle_impl& handle) {
	ASSERT(n_cols > 0,
			"ridgeSVD: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"ridgeSVD: number of rows cannot be less than two");
        auto allocator = handle.getDeviceAllocator();
        auto stream = handle.getStream();
        auto cusolverH = handle.getcusolverDnHandle();
        auto cublasH = handle.getCublasHandle();

	int U_len = n_rows * n_cols;
	int V_len = n_cols * n_cols;

        device_buffer<math_t> U(allocator, stream, U_len);
        device_buffer<math_t> V(allocator, stream, V_len);
        device_buffer<math_t> S(allocator, stream, n_cols);

	LinAlg::svdQR(A, n_rows, n_cols, S.data(), U.data(), V.data(), true, true, true,
                      cusolverH, cublasH, allocator, stream);
	ridgeSolve(S.data(), V.data(), U.data(), n_rows, n_cols, b, alpha, n_alpha, w,
                   cusolverH, cublasH);
}

template<typename math_t>
void ridgeEig(math_t *A, int n_rows, int n_cols, math_t *b, math_t *alpha,
		int n_alpha, math_t *w, const cumlHandle_impl& handle) {
	ASSERT(n_cols > 1,
			"ridgeEig: number of columns cannot be less than two");
	ASSERT(n_rows > 1,
			"ridgeEig: number of rows cannot be less than two");
        auto allocator = handle.getDeviceAllocator();
        auto stream = handle.getStream();
        auto cusolverH = handle.getcusolverDnHandle();
        auto cublasH = handle.getCublasHandle();

	int U_len = n_rows * n_cols;
	int V_len = n_cols * n_cols;

        device_buffer<math_t> U(allocator, stream, U_len);
        device_buffer<math_t> V(allocator, stream, V_len);
        device_buffer<math_t> S(allocator, stream, n_cols);

	LinAlg::svdEig(A, n_rows, n_cols, S.data(), U.data(), V.data(), true, cublasH, cusolverH,
                       allocator, stream);
	ridgeSolve(S.data(), V.data(), U.data(), n_rows, n_cols, b, alpha, n_alpha, w, cusolverH,
                   cublasH);
}

template<typename math_t>
void ridgeFit(const cumlHandle_impl& handle, math_t *input, int n_rows, int n_cols,
              math_t *labels, math_t *alpha, int n_alpha, math_t *coef, math_t *intercept,
              bool fit_intercept, bool normalize, int algo = 0) {
    auto allocator = handle.getDeviceAllocator();
    auto stream = handle.getStream();
    auto cublas_handle = handle.getCublasHandle();
    auto cusolver_handle = handle.getcusolverDnHandle();

	ASSERT(n_cols > 0,
			"ridgeFit: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"ridgeFit: number of rows cannot be less than two");

        device_buffer<math_t> mu_input(allocator, stream);
        device_buffer<math_t> norm2_input(allocator, stream);
        device_buffer<math_t> mu_labels(allocator, stream);

	if (fit_intercept) {
            mu_input.resize(n_cols, stream);
            mu_labels.resize(1, stream);
            if (normalize) {
                norm2_input.resize(n_cols, stream);
            }
            preProcessData(input, n_rows, n_cols, labels, intercept, mu_input.data(),
                           mu_labels.data(), norm2_input.data(), fit_intercept,
                           normalize, cublas_handle, cusolver_handle, stream);
	}

	if (algo == 0 || n_cols == 1) {
		ridgeSVD(input, n_rows, n_cols, labels, alpha, n_alpha, coef,
                         handle);
	} else if (algo == 1) {
		ridgeEig(input, n_rows, n_cols, labels, alpha, n_alpha, coef,
                         handle);
	} else if (algo == 2) {
		ASSERT(false,
				"ridgeFit: no algorithm with this id has been implemented");
	} else {
		ASSERT(false,
				"ridgeFit: no algorithm with this id has been implemented");
	}

	if (fit_intercept) {
            postProcessData(input, n_rows, n_cols, labels, coef, intercept, mu_input.data(),
                            mu_labels.data(), norm2_input.data(), fit_intercept, normalize,
                            cublas_handle, cusolver_handle, stream);
	} else {
		*intercept = math_t(0);
	}
}

template<typename math_t>
void ridgePredict(const cumlHandle_impl& handle, const math_t *input, int n_rows, int n_cols,
                  const math_t *coef, math_t intercept, math_t *preds) {

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

        auto stream = handle.getStream();
	math_t alpha = math_t(1);
	math_t beta = math_t(0);
	LinAlg::gemm(input, n_rows, n_cols, coef, preds, n_rows, 1, CUBLAS_OP_N,
                     CUBLAS_OP_N, alpha, beta, handle.getCublasHandle(), stream);
	LinAlg::addScalar(preds, preds, intercept, n_rows, stream);
}

/** @} */
}; // end namespace GLM
}; // end namespace ML
