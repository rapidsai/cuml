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
#include <cuda_utils.h>
#include <linalg/gemm.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/eltwise.h>
#include <linalg/unary_op.h>
#include <linalg/cublas_wrappers.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include "glm/preprocess.h"
#include "shuffle.h"
#include <functions/penalty.h>
#include <functions/softThres.h>

namespace ML {
namespace Solver {

using namespace MLCommon;

template<typename math_t>
void cdFit(math_t *input,
		    int n_rows,
		    int n_cols,
		    math_t *labels,
		    math_t *coef,
		    math_t *intercept,
		    bool fit_intercept,
		    int epochs,
		    ML::loss_funct loss,
		    Functions::penalty penalty,
		    math_t alpha,
		    math_t l1_ratio,
		    bool shuffle,
		    math_t tol,
		    int n_iter_no_change,
		    cublasHandle_t cublas_handle,
		    cusolverDnHandle_t cusolver_handle) {

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

	math_t *mu_input = NULL;
	math_t *mu_labels = NULL;
	math_t *norm2_input = NULL;
	math_t *pred = NULL;
	math_t *squared = NULL;

	allocate(pred, n_rows);
	allocate(squared, n_cols);

	if (fit_intercept) {
		allocate(mu_input, n_cols);
		allocate(mu_labels, 1);

		GLM::preProcessData(input, n_rows, n_cols, labels, intercept, mu_input,
				mu_labels, norm2_input, fit_intercept, false, cublas_handle,
				cusolver_handle);
	}

	std::vector<int> rand_indices(n_cols);
	std::mt19937 g(rand());
	initShuffle(rand_indices, g);

	if (penalty == Functions::penalty::L1)
		alpha = alpha * n_rows;

	LinAlg::colNorm(squared, input, n_cols, n_rows, LinAlg::L2Norm,
	               []__device__(math_t v){ return v; });

	for (int i = 0; i < epochs; i++) {
		if (i > 0 && shuffle) {
			Solver::shuffle(rand_indices, g);
		}

		for (int j = 0; j < n_cols; j++) {
			math_t *coef_loc = coef + rand_indices[j];
			Matrix::setValue(coef_loc, coef_loc, math_t(0), 1);

			LinAlg::gemm(input, n_rows, n_cols, coef, pred, n_rows, 1, CUBLAS_OP_N,
						CUBLAS_OP_N, cublas_handle);
			LinAlg::subtract(pred, labels, pred, n_rows);

			math_t *input_col_loc = input + (rand_indices[j] * n_rows);
			LinAlg::gemm(input_col_loc, n_rows, 1, pred, coef_loc, 1, 1, CUBLAS_OP_T,
									CUBLAS_OP_N, cublas_handle);

			if (penalty == Functions::penalty::L1)
				Functions::softThres(coef_loc, coef_loc, alpha, 1);

		}

		if (tol > math_t(0)) {

		}
	}

	if (fit_intercept) {
		GLM::postProcessData(input, n_rows, n_cols, labels, coef, intercept,
				mu_input, mu_labels, norm2_input, fit_intercept, false,
				cublas_handle, cusolver_handle);

		if (mu_input != NULL)
			CUDA_CHECK(cudaFree(mu_input));
		if (mu_labels != NULL)
			CUDA_CHECK(cudaFree(mu_labels));
	} else {
		*intercept = math_t(0);
	}

	if (pred != NULL)
		CUDA_CHECK(cudaFree(pred));

	if (squared != NULL)
		CUDA_CHECK(cudaFree(squared));

}

template<typename math_t>
void cdPredict(const math_t *input, int n_rows, int n_cols, const math_t *coef,
		math_t intercept, math_t *preds, ML::loss_funct loss, cublasHandle_t cublas_handle) {

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");


}

/** @} */
}
;
}
;
// end namespace ML
