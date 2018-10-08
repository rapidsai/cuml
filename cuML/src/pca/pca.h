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

#include <linalg/eltwise.h>
#include <linalg/eig.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/transpose.h>
#include <stats/cov.h>
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <cuda_utils.h>
#include <matrix/matrix.h>
#include <matrix/math.h>
#include "ml_utils.h"
#include "tsvd/tsvd.h"

namespace ML {

using namespace MLCommon;

// TODO: Implement for the case prms.trans_input = false
// TODO: Check sign flip algorithm
/**
 * @brief perform fit operation for the pca. Generates eigenvectors, explained vars, singular vals, etc.
 * @input param input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @output param components: the principal components of the input data. Size n_cols * n_components.
 * @output param explained_var: explained variances (eigenvalues) of the principal components. Size n_components * 1.
 * @output param explained_var_ratio: the ratio of the explained variance and total variance. Size n_components * 1.
 * @output param singular_vals: singular values of the data. Size n_components * 1
 * @output param mu: mean of all the features (all the columns in the data). Size n_cols * 1.
 * @output param noise_vars: variance of the noise. Size 1 * 1 (scalar).
 * @input param prms: data structure that includes all the parameters from input size to algorithm.
 * @input param cublas_handle: cublas handle
 * @input param cusolver_handle: cusolver handle
 */
template<typename math_t>
void pcaFit(math_t *input, math_t *components, math_t *explained_var,
		math_t *explained_var_ratio, math_t *singular_vals, math_t *mu,
		math_t *noise_vars, paramsPCA prms, cublasHandle_t cublas_handle,
		cusolverDnHandle_t cusolver_handle) {

	ASSERT(prms.n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than two");
	ASSERT(prms.n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");
	ASSERT(prms.n_components > 0,
			"Parameter n_components: number of components cannot be less than one");

	if (prms.n_components > prms.n_cols)
		prms.n_components = prms.n_cols;

	Stats::mean(mu, input, prms.n_cols, prms.n_rows, true, false);

	if (prms.algorithm == solver::RANDOMIZED) {
		Stats::meanCenter(input, mu, prms.n_cols, prms.n_rows, false);
		calCompExpVarsSvd(input, components, singular_vals, explained_var,
				explained_var_ratio, prms, cusolver_handle, cublas_handle);
	} else {
		math_t *cov;
		int len = prms.n_cols * prms.n_cols;
		allocate(cov, len);

		Stats::cov(cov, input, mu, prms.n_cols, prms.n_rows, true, false, true,
				cublas_handle);
		calCompExpVarsEig(cov, components, explained_var, explained_var_ratio,
				prms, cusolver_handle, cublas_handle);

		math_t scalar = (prms.n_rows - 1);
		Matrix::seqRoot(explained_var, singular_vals, scalar,
				prms.n_components);

		if (cov)
			CUDA_CHECK(cudaFree(cov));
	}

	Stats::meanAdd(input, mu, prms.n_cols, prms.n_rows, false);
}

/**
 * @brief perform fit and transform operations for the pca. Generates transformed data, eigenvectors, explained vars, singular vals, etc.
 * @input param input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @output param trans_input: the transformed data. Size n_rows * n_components.
 * @output param components: the principal components of the input data. Size n_cols * n_components.
 * @output param explained_var: explained variances (eigenvalues) of the principal components. Size n_components * 1.
 * @output param explained_var_ratio: the ratio of the explained variance and total variance. Size n_components * 1.
 * @output param singular_vals: singular values of the data. Size n_components * 1
 * @output param mu: mean of all the features (all the columns in the data). Size n_cols * 1.
 * @output param noise_vars: variance of the noise. Size 1 * 1 (scalar).
 * @input param prms: data structure that includes all the parameters from input size to algorithm.
 * @input param cublas_handle: cublas handle
 * @input param cusolver_handle: cusolver handle
 */
template<typename math_t>
void pcaFitTransform(math_t *input, math_t *trans_input, math_t *components,
		math_t *explained_var, math_t *explained_var_ratio,
		math_t *singular_vals, math_t *mu, math_t *noise_vars, paramsPCA prms,
		cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle) {

	pcaFit(input, components, explained_var, explained_var_ratio, singular_vals,
			mu, noise_vars, prms, cublas_handle, cusolver_handle);

	pcaTransform(input, components, trans_input, singular_vals, mu, prms,
			cublas_handle);

	signFlip(trans_input, prms.n_rows, prms.n_components, components, prms.n_cols);
}

// TODO: implement pcaGetCovariance function
template<typename math_t>
void pcaGetCovariance() {
	ASSERT(false, "pcaGetCovariance: will be implemented!");
}

// TODO: implement pcaGetPrecision function
template<typename math_t>
void pcaGetPrecision() {
	ASSERT(false, "pcaGetPrecision: will be implemented!");
}

/**
 * @brief performs inverse transform operation for the pca. Transforms the transformed data back to original data.
 * @input param trans_input: the data is fitted to PCA. Size n_rows x n_components.
 * @input param components: transpose of the principal components of the input data. Size n_components * n_cols.
 * @input param singular_vals: singular values of the data. Size n_components * 1
 * @input param mu: mean of features (every column).
 * @output param input: the data is fitted to PCA. Size n_rows x n_cols.
 * @input param prms: data structure that includes all the parameters from input size to algorithm.
 * @input param cublas_handle: cublas handle
 */

template<typename math_t>
void pcaInverseTransform(math_t *trans_input, math_t *components,
		math_t *singular_vals, math_t *mu, math_t *input, paramsPCA prms,
		cublasHandle_t cublas_handle) {

	ASSERT(prms.n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than two");
	ASSERT(prms.n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");
	ASSERT(prms.n_components > 0,
			"Parameter n_components: number of components cannot be less than one");

	if (prms.whiten) {
		math_t scalar = math_t(1 / sqrt(prms.n_rows - 1));
		LinAlg::scalarMultiply(components, components, scalar,
				prms.n_rows * prms.n_components);
		Matrix::matrixVectorBinaryMultSkipZero(components, singular_vals,
				prms.n_rows, prms.n_components, true);
	}

	tsvdInverseTransform(trans_input, components, input, prms, cublas_handle);
	Stats::meanAdd(input, mu, prms.n_cols, prms.n_rows, false);

	if (prms.whiten) {
		Matrix::matrixVectorBinaryDivSkipZero(components, singular_vals,
				prms.n_rows, prms.n_components, true);
		math_t scalar = math_t(sqrt(prms.n_rows - 1));
		LinAlg::scalarMultiply(components, components, scalar,
				prms.n_rows * prms.n_components);
	}
}

// TODO: implement pcaScore function
template<typename math_t>
void pcaScore() {
	ASSERT(false, "pcaScore: will be implemented!");
}

// TODO: implement pcaScoreSamples function
template<typename math_t>
void pcaScoreSamples() {
	ASSERT(false, "pcaScoreSamples: will be implemented!");
}

/**
 * @brief performs transform operation for the pca. Transforms the data to eigenspace.
 * @input param input: the data is transformed. Size n_rows x n_components.
 * @input param components: principal components of the input data. Size n_cols * n_components.
 * @output param trans_input:  the transformed data. Size n_rows * n_components.
 * @input param singular_vals: singular values of the data. Size n_components * 1.
 * @input param prms: data structure that includes all the parameters from input size to algorithm.
 * @input param cublas_handle: cublas handle
 */
template<typename math_t>
void pcaTransform(math_t *input, math_t *components, math_t *trans_input,
		math_t *singular_vals, math_t *mu, paramsPCA prms,
		cublasHandle_t cublas_handle) {

	ASSERT(prms.n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than two");
	ASSERT(prms.n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");
	ASSERT(prms.n_components > 0,
			"Parameter n_components: number of components cannot be less than one");

	if (prms.whiten) {
		math_t scalar = math_t(sqrt(prms.n_rows - 1));
		LinAlg::scalarMultiply(components, components, scalar,
				prms.n_rows * prms.n_components);
		Matrix::matrixVectorBinaryDivSkipZero(components, singular_vals,
				prms.n_rows, prms.n_components, true);
	}

	Stats::meanCenter(input, mu, prms.n_cols, prms.n_rows, false);
	tsvdTransform(input, components, trans_input, prms, cublas_handle);
	Stats::meanAdd(input, mu, prms.n_cols, prms.n_rows, false);

	if (prms.whiten) {
		Matrix::matrixVectorBinaryMultSkipZero(components, singular_vals,
				prms.n_rows, prms.n_components, true);
		math_t scalar = math_t(1 / sqrt(prms.n_rows - 1));
		LinAlg::scalarMultiply(components, components, scalar,
				prms.n_rows * prms.n_components);
	}

}

/** @} */

}
;
// end namespace ML
