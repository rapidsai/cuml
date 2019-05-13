/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#include <cuda_utils.h>
#include <linalg/gemm.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/multiply.h>
#include <linalg/eltwise.h>
#include <linalg/unary_op.h>
#include <linalg/cublas_wrappers.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include "glm/preprocess.h"
#include "shuffle.h"
#include <functions/penalty.h>
#include <functions/softThres.h>
#include <functions/linearReg.h>
#include "common/cumlHandle.hpp"

namespace ML {
namespace Solver {

using namespace MLCommon;

/**
 * Fits a linear, lasso, and elastic-net regression model using Coordinate Descent solver
 * @param cumlHandle_impl
 *        Reference of cumlHandle
 * @param input
 *        pointer to an array in column-major format (size of n_rows, n_cols)
 * @param n_rows
 *        n_samples or rows in input
 * @param n_cols
 *        n_features or columns in X
 * @param labels
 *        pointer to an array for labels (size of n_rows)
 * @param coef
 *        pointer to an array for coefficients (size of n_cols). This will be filled with coefficients
 *        once the function is executed.
 * @param intercept
 *        pointer to a scalar for intercept. This will be filled
 *        once the function is executed
 * @param fit_intercept
 *        boolean parameter to control if the intercept will be fitted or not
 * @param normalize
 *        boolean parameter to control if the data will be normalized or not
 * @param loss
 *        enum to use different loss functions. Only linear regression loss functions is supported right now.
 * @param alpha
 *        L1 parameter
 * @param l1_ratio
 *        ratio of alpha will be used for L1. (1 - l1_ratio) * alpha will be used for L2.
 * @param shuffle
 *        boolean parameter to control whether coordinates will be picked randomly or not.
 * @param tol
 *        tolerance to stop the solver
 * @param stream
 *        cuda stream
 */
template<typename math_t>
void cdFit(const cumlHandle_impl& handle, math_t *input, int n_rows, int n_cols,
		math_t *labels, math_t *coef, math_t *intercept, bool fit_intercept,
		bool normalize, int epochs, ML::loss_funct loss, math_t alpha,
		math_t l1_ratio, bool shuffle, math_t tol, cudaStream_t stream) {

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");
	ASSERT(loss == ML::loss_funct::SQRD_LOSS,
			"Parameter loss: Only SQRT_LOSS function is supported for now");

	cublasHandle_t cublas_handle = handle.getCublasHandle();
	cusolverDnHandle_t cusolver_handle = handle.getcusolverDnHandle();

	auto allocator = handle.getDeviceAllocator();
	device_buffer<math_t> pred(allocator, stream, n_rows);
	device_buffer<math_t> residual(allocator, stream, n_rows);
	device_buffer<math_t> squared(allocator, stream, n_cols);
	device_buffer<math_t> mu_input(allocator, stream, 0);
	device_buffer<math_t> mu_labels(allocator, stream, 0);
	device_buffer<math_t> norm2_input(allocator, stream, 0);

	std::vector<math_t> h_coef(n_cols, math_t(0));

	if (fit_intercept) {
		mu_input.resize(n_cols, stream);
		mu_labels.resize(1, stream);
		if (normalize) {
			norm2_input.resize(n_cols, stream);
		}

		GLM::preProcessData(handle, input, n_rows, n_cols, labels,
				intercept, mu_input.data(), mu_labels.data(), norm2_input.data(), fit_intercept,
				normalize, stream);
	}

	std::vector<int> ri(n_cols);
	std::mt19937 g(rand());
	initShuffle(ri, g);

	math_t l2_alpha = (1 - l1_ratio) * alpha * n_rows;
	alpha = l1_ratio * alpha * n_rows;

	if (normalize) {
		math_t scalar = math_t(1.0) + l2_alpha;
		Matrix::setValue(squared.data(), squared.data(), scalar, n_cols, stream);
	} else {
		LinAlg::colNorm(squared.data(), input, n_cols, n_rows, LinAlg::L2Norm, false,
				stream);
		LinAlg::addScalar(squared.data(), squared.data(), l2_alpha, n_cols, stream);
	}

	copy(residual.data(), labels, n_rows, stream);

	for (int i = 0; i < epochs; i++) {
		if (i > 0 && shuffle) {
			Solver::shuffle(ri, g);
		}

		math_t coef_max = 0.0;
		math_t d_coef_max = 0.0;
		math_t coef_prev = 0.0;

		for (int j = 0; j < n_cols; j++) {
			int ci = ri[j];
			math_t *coef_loc = coef + ci;
			math_t *squared_loc = squared.data() + ci;
			math_t *input_col_loc = input + (ci * n_rows);

			LinAlg::multiplyScalar(pred.data(), input_col_loc, h_coef[ci], n_rows,
					stream);
			LinAlg::add(residual.data(), residual.data(), pred.data(), n_rows, stream);
			LinAlg::gemm(input_col_loc, n_rows, 1, residual.data(), coef_loc, 1, 1,
					CUBLAS_OP_T, CUBLAS_OP_N, cublas_handle, stream);

			if (l1_ratio > math_t(0.0))
				Functions::softThres(coef_loc, coef_loc, alpha, 1, stream);

			LinAlg::eltwiseDivideCheckZero(coef_loc, coef_loc, squared_loc, 1,
					stream);

			coef_prev = h_coef[ci];
			updateHost(&(h_coef[ci]), coef_loc, 1, stream);
			math_t diff = abs(coef_prev - h_coef[ci]);

			if (diff > d_coef_max)
				d_coef_max = diff;

			if (abs(h_coef[ci]) > coef_max)
				coef_max = abs(h_coef[ci]);

			LinAlg::multiplyScalar(pred.data(), input_col_loc, h_coef[ci], n_rows,
					stream);
			LinAlg::subtract(residual.data(), residual.data(), pred.data(), n_rows, stream);
		}

		bool flag_continue = true;
		if (coef_max == math_t(0)) {
			flag_continue = false;
		}

		if ((d_coef_max / coef_max) < tol) {
			flag_continue = false;
		}

		if (!flag_continue) {
			break;
		}
	}

	if (fit_intercept) {
		GLM::postProcessData(handle, input, n_rows, n_cols, labels,
				coef, intercept, mu_input.data(), mu_labels.data(), norm2_input.data(),
				fit_intercept, normalize, stream);

	} else {
		*intercept = math_t(0);
	}

}

/**
 * Fits a linear, lasso, and elastic-net regression model using Coordinate Descent solver
 * @param input
 *        pointer to an array in column-major format (size of n_rows, n_cols)
 * @param n_rows
 *        n_samples or rows in input
 * @param n_cols
 *        n_features or columns in X
 * @param coef
 *        pointer to an array for coefficients (size of n_cols). Calculated in cdFit function.
 * @param intercept
 *        intercept value calculated in cdFit function
 * @param preds
 *        pointer to an array for predictions (size of n_rows). This will be fitted once functions is executed.
 * @param loss
 *        enum to use different loss functions. Only linear regression loss functions is supported right now.
 * @param stream
 *        cuda stream
 * @param cublas_handle
 *        cublas handle
 */
template<typename math_t>
void cdPredict(const cumlHandle_impl& handle, const math_t *input, int n_rows,
		int n_cols, const math_t *coef, math_t intercept, math_t *preds,
		ML::loss_funct loss, cudaStream_t stream) {

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");
	ASSERT(loss == ML::loss_funct::SQRD_LOSS,
			"Parameter loss: Only SQRT_LOSS function is supported for now");

	cublasHandle_t cublas_handle = handle.getCublasHandle();
	Functions::linearRegH(input, n_rows, n_cols, coef, preds, intercept,
			cublas_handle, stream);

}

/** @} */
}
;
}
;
// end namespace ML
