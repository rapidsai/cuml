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

#include "tsvd.h"
#include "tsvd_c.h"


namespace ML {

using namespace MLCommon;

void tsvdFit(float *input, float *components, float *explained_var,
		float *explained_var_ratio, float *singular_vals, paramsTSVD prms) {
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	tsvdFit(input, components, explained_var, explained_var_ratio,
			singular_vals, prms, cublas_handle, cusolver_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
}

void tsvdFit(double *input, double *components, double *explained_var,
		double *explained_var_ratio, double *singular_vals, paramsTSVD prms) {
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	tsvdFit(input, components, explained_var, explained_var_ratio,
			singular_vals, prms, cublas_handle, cusolver_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
}

void tsvdFitTransform(float *input, float *trans_input, float *components,
		float *explained_var, float *explained_var_ratio, float *singular_vals,
		paramsTSVD prms) {
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	tsvdFitTransform(input, trans_input, components, explained_var,
			explained_var_ratio, singular_vals, prms, cublas_handle,
			cusolver_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

}

void tsvdFitTransform(double *input, double *trans_input, double *components,
		double *explained_var, double *explained_var_ratio,
		double *singular_vals, paramsTSVD prms) {
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	cusolverDnHandle_t cusolver_handle = NULL;
	CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

	tsvdFitTransform(input, trans_input, components, explained_var,
			explained_var_ratio, singular_vals, prms, cublas_handle,
			cusolver_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

}

void tsvdTransform(float *input, float *components, float *trans_input,
		paramsTSVD prms) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	tsvdTransform(input, components, trans_input, prms, cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

void tsvdTransform(double *input, double *components, double *trans_input,
		paramsTSVD prms) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	tsvdTransform(input, components, trans_input, prms, cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

void tsvdInverseTransform(float *trans_input, float *components, float *input,
		paramsTSVD prms) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	tsvdInverseTransform(trans_input, components, input, prms, cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

void tsvdInverseTransform(double *trans_input, double *components,
		double *input, paramsTSVD prms) {

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	tsvdInverseTransform(trans_input, components, input, prms, cublas_handle);

	CUBLAS_CHECK(cublasDestroy(cublas_handle));

}

/** @} */

}
;
// end namespace ML
