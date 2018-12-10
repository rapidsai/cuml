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

#include "ols.h"
#include "glm_c.h"

namespace ML {
namespace GLM {

using namespace MLCommon;

void olsFit(float *input, int n_rows, int n_cols, float *labels, float *coef,
		float *intercept, bool fit_intercept, bool normalize, int algo) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	olsFit(input, n_rows, n_cols, labels, coef, intercept, fit_intercept,
			normalize, cublas_handle, cusolver_handle, algo);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

}

void olsFit(double *input, int n_rows, int n_cols, double *labels, double *coef,
		double *intercept, bool fit_intercept, bool normalize, int algo) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	olsFit(input, n_rows, n_cols, labels, coef, intercept, fit_intercept,
			normalize, cublas_handle, cusolver_handle, algo);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

}

void olsPredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	olsPredict(input, n_rows, n_cols, coef, intercept, preds, cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));

}

void olsPredict(const double *input, int n_rows, int n_cols, const double *coef,
		double intercept, double *preds) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	olsPredict(input, n_rows, n_cols, coef, intercept, preds, cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));

}

}
}
