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

#include <cuda_utils.h>
#include <stats/mean.h>
#include <stats/sum.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/eltwise.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/transpose.h>
#include <linalg/gemm.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include <linalg/add.h>
#include "penalty.h"


namespace MLCommon {
namespace Functions {

template<typename math_t>
void linearRegH(const math_t *input, int n_rows, int n_cols,
		 const math_t *coef, math_t *pred, math_t intercept,
		 cublasHandle_t cublas_handle, cudaStream_t stream) {

	LinAlg::gemm(input, n_rows, n_cols, coef, pred, n_rows, 1, CUBLAS_OP_N,
			CUBLAS_OP_N, cublas_handle, stream);

	if (intercept != math_t(0))
		LinAlg::addScalar(pred, pred, intercept, n_rows, stream);

}

template<typename math_t>
void linearRegLossGrads(math_t *input, int n_rows, int n_cols,
		const math_t *labels, const math_t *coef, math_t *grads, penalty pen,
		math_t alpha, math_t l1_ratio, cublasHandle_t cublas_handle,
		std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream) {

	device_buffer<math_t> labels_pred(allocator, stream, n_rows);
	device_buffer<math_t> input_t(allocator, stream, n_rows * n_cols);

	linearRegH(input, n_rows, n_cols, coef, labels_pred.data(), math_t(0), cublas_handle, stream);

	LinAlg::subtract(labels_pred.data(), labels_pred.data(), labels, n_rows, stream);

	// TODO: implement a matrixVectorBinaryMult that runs on rows rather than columns.
	LinAlg::transpose(input, input_t.data(), n_rows, n_cols, cublas_handle, stream);
	Matrix::matrixVectorBinaryMult(input_t.data(), labels_pred.data(), n_cols, n_rows, false, true, stream);
	LinAlg::transpose(input_t.data(), input, n_cols, n_rows, cublas_handle, stream);

	Stats::mean(grads, input, n_cols, n_rows, false, false, stream);
	LinAlg::scalarMultiply(grads, grads, math_t(2), n_cols, stream);

	math_t *pen_grads = NULL;

	if (pen != penalty::NONE)
		allocate(pen_grads, n_cols);

	if (pen == penalty::L1) {
		lassoGrad(pen_grads, coef, n_cols, alpha, stream);
	} else if (pen == penalty::L2) {
		ridgeGrad(pen_grads, coef, n_cols, alpha, stream);
	} else if (pen == penalty::ELASTICNET) {
		elasticnetGrad(pen_grads, coef, n_cols, alpha, l1_ratio, stream);
	}

	if (pen != penalty::NONE) {
	    LinAlg::add(grads, grads, pen_grads, n_cols, stream);
	    if (pen_grads != NULL)
	        CUDA_CHECK(cudaFree(pen_grads));
	}

}


template<typename math_t>
void linearRegLoss(math_t *input, int n_rows, int n_cols,
		const math_t *labels, const math_t *coef, math_t *loss, penalty pen,
		math_t alpha, math_t l1_ratio, cublasHandle_t cublas_handle,
		std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream) {

	device_buffer<math_t> labels_pred(allocator, stream, n_rows);

	linearRegH(input, n_rows, n_cols, coef, labels_pred.data(), math_t(0), cublas_handle, stream);

	LinAlg::subtract(labels_pred.data(), labels, labels_pred.data(), n_rows, stream);
	Matrix::power(labels_pred.data(), n_rows, stream);
	Stats::mean(loss, labels_pred.data(), 1, n_rows, false, false, stream);

	math_t *pen_val = NULL;

    if (pen != penalty::NONE)
	    allocate(pen_val, 1);

	if (pen == penalty::L1) {
		lasso(pen_val, coef, n_cols, alpha, stream);
	} else if (pen == penalty::L2) {
		ridge(pen_val, coef, n_cols, alpha, stream);
	} else if (pen == penalty::ELASTICNET) {
		elasticnet(pen_val, coef, n_cols, alpha, l1_ratio, stream);
	}

	if (pen != penalty::NONE) {
	    LinAlg::add(loss, loss, pen_val, 1, stream);
	    if (pen_val != NULL)
	        CUDA_CHECK(cudaFree(pen_val));
	}

}

/** @} */
}
;
}
;
// end namespace ML
