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
#include <linalg/subtract.h>
#include <linalg/eltwise.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/transpose.h>
#include <linalg/gemm.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include <linalg/add.h>
#include "penalty.h"
#include "sigmoid.h"
#include <linalg/binary_op.h>


namespace MLCommon {
namespace Functions {

template<typename math_t>
void logisticRegH(const math_t *input, int n_rows, int n_cols,
		 const math_t *coef, math_t *pred, math_t intercept,
		 cublasHandle_t cublas_handle) {

	LinAlg::gemm(input, n_rows, n_cols, coef, pred, n_rows, 1, CUBLAS_OP_N,
			CUBLAS_OP_N, cublas_handle);

	if (intercept != math_t(0))
		LinAlg::addScalar(pred, pred, intercept, n_rows);

	sigmoid(pred, pred, n_rows);
}

template<typename math_t>
void logisticRegLossGrads(math_t *input, int n_rows, int n_cols,
		const math_t *labels, const math_t *coef, math_t *grads, penalty pen,
		math_t alpha, math_t l1_ratio, cublasHandle_t cublas_handle) {

	math_t *labels_pred = NULL;
	allocate(labels_pred, n_rows);

	math_t *input_t = NULL;
	allocate(input_t, n_rows * n_cols);

	logisticRegH(input, n_rows, n_cols, coef, labels_pred, math_t(0), cublas_handle);

	LinAlg::subtract(labels_pred, labels_pred, labels, n_rows);

	// TODO: implement a matrixVectorBinaryMult that runs on rows rather than columns.
	LinAlg::transpose(input, input_t, n_rows, n_cols, cublas_handle);
	Matrix::matrixVectorBinaryMult(input_t, labels_pred, n_cols, n_rows, false, true);
	LinAlg::transpose(input_t, input, n_cols, n_rows, cublas_handle);

	Stats::mean(grads, input, n_cols, n_rows, false, false);

	math_t *pen_grads = NULL;

	if (pen != penalty::NONE)
		allocate(pen_grads, n_cols);

	if (pen == penalty::L1) {
		lassoGrad(pen_grads, coef, n_cols, alpha);
	} else if (pen == penalty::L2) {
		ridgeGrad(pen_grads, coef, n_cols, alpha);
	} else if (pen == penalty::ELASTICNET) {
		elasticnetGrad(pen_grads, coef, n_cols, alpha, l1_ratio);
	}

	if (pen != penalty::NONE) {
	    LinAlg::add(grads, grads, pen_grads, n_cols);
	    if (pen_grads != NULL)
	        CUDA_CHECK(cudaFree(pen_grads));
	}

	if (labels_pred != NULL)
	    CUDA_CHECK(cudaFree(labels_pred));

	if (input_t != NULL)
		CUDA_CHECK(cudaFree(input_t));
}

template <typename T>
void logLoss(T *out, T *label, T *label_pred, int len);


template <>
inline void logLoss(float *out, float *label, float *label_pred, int len) {
	LinAlg::binaryOp(out, label, label_pred, len, [] __device__ (float y, float y_pred) {
	                                           return -y * logf(y_pred) - (1 - y) * logf(1 - y_pred);
	                                       });
}

template <>
inline void logLoss(double *out, double *label, double *label_pred, int len) {
	LinAlg::binaryOp(out, label, label_pred, len, [] __device__ (double y, double y_pred) {
	                                           return -y * log(y_pred) - (1 - y) * logf(1 - y_pred);
	                                       });
}

template<typename math_t>
void logisticRegLoss(math_t *input, int n_rows, int n_cols,
		math_t *labels, const math_t *coef, math_t *loss, penalty pen,
		math_t alpha, math_t l1_ratio, cublasHandle_t cublas_handle) {

	math_t *labels_pred = NULL;
	allocate(labels_pred, n_rows);

	logisticRegH(input, n_rows, n_cols, coef, labels_pred, math_t(0), cublas_handle);
	logLoss(labels_pred, labels, labels_pred, n_rows);

	Stats::mean(loss, labels_pred, 1, n_rows, false, false);

	math_t *pen_val = NULL;

    if (pen != penalty::NONE)
	    allocate(pen_val, 1);

	if (pen == penalty::L1) {
		lasso(pen_val, coef, n_cols, alpha);
	} else if (pen == penalty::L2) {
		ridge(pen_val, coef, n_cols, alpha);
	} else if (pen == penalty::ELASTICNET) {
		elasticnet(pen_val, coef, n_cols, alpha, l1_ratio);
	}

	if (pen != penalty::NONE) {
	    LinAlg::add(loss, loss, pen_val, 1);
	    if (pen_val != NULL)
	        CUDA_CHECK(cudaFree(pen_val));
	}

	if (labels_pred != NULL)
	    CUDA_CHECK(cudaFree(labels_pred));

}

/** @} */
}
;
}
;
// end namespace ML
