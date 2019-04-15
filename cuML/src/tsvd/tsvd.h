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

#include <linalg/binary_op.h>
#include <linalg/eltwise.h>
#include <linalg/eig.h>
#include <linalg/rsvd.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/transpose.h>
#include <linalg/gemm.h>
#include <cuda_utils.h>
#include <matrix/matrix.h>
#include <matrix/math.h>
#include <stats/stddev.h>
#include <stats/mean.h>
#include <stats/sum.h>
#include "ml_utils.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"

namespace ML {

using namespace MLCommon;

template<typename math_t>
void calCompExpVarsSvd(math_t *in, math_t *components, math_t *singular_vals,
		math_t *explained_vars, math_t *explained_var_ratio, paramsTSVD prms,
		cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle,
    cudaStream_t stream) {
	int diff = prms.n_cols - prms.n_components;
	math_t ratio = math_t(diff) / math_t(prms.n_cols);
	ASSERT(ratio >= math_t(0.2),
			"Number of components should be less than at least 80 percent of the number of features");

	int p = int(math_t(0.1) * math_t(prms.n_cols));
	// int p = int(math_t(prms.n_cols) / math_t(4));
	ASSERT(p >= 5,
			"RSVD should be used where the number of columns are at least 50");

	int total_random_vecs = prms.n_components + p;
	ASSERT(total_random_vecs < prms.n_cols,
			"RSVD should be used where the number of columns are at least 50");

	math_t *components_temp;
	allocate(components_temp, prms.n_cols, prms.n_components);
	math_t *left_eigvec;
	LinAlg::rsvdFixedRank(in, prms.n_rows, prms.n_cols, singular_vals,
			left_eigvec, components_temp, prms.n_components, p, true, false,
			true, false, (math_t) prms.tol, prms.n_iterations, cusolver_handle,
			cublas_handle);

	LinAlg::transpose(components_temp, components, prms.n_cols,
			prms.n_components, cublas_handle, stream);
	Matrix::power(singular_vals, explained_vars, math_t(1), prms.n_components);
  auto mgr = makeDefaultAllocator();
  Matrix::ratio(explained_vars, explained_var_ratio, prms.n_components, mgr);

	if (components_temp)
		CUDA_CHECK(cudaFree(components_temp));

}

template<typename math_t>
void calEig(const cumlHandle_impl& handle, math_t *in, math_t *components,
            math_t *explained_var, paramsTSVD prms) {
    auto stream = handle.getStream();
    auto cusolver_handle = handle.getcusolverDnHandle();
    auto allocator = handle.getDeviceAllocator();

	if (prms.algorithm == solver::COV_EIG_JACOBI) {
		LinAlg::eigJacobi(in, prms.n_cols, prms.n_cols, components,
				explained_var, (math_t) prms.tol, prms.n_iterations,
                                  cusolver_handle, stream, allocator);
	} else {
		LinAlg::eigDC(in, prms.n_cols, prms.n_cols, components, explained_var,
                              cusolver_handle, stream, allocator);
	}

	Matrix::colReverse(components, prms.n_cols, prms.n_cols, stream);
	LinAlg::transpose(components, prms.n_cols, stream);

	Matrix::rowReverse(explained_var, prms.n_cols, 1, stream);
}

/**
 * @defgroup sign flip for PCA and tSVD. This is used to stabilize the sign of column major eigen vectors
 * @param input: input matrix that will be used to determine the sign.
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param components: components matrix.
 * @param n_cols_comp: number of columns of components matrix
 * @{
 */
template<typename math_t>
void signFlip(math_t *input, int n_rows, int n_cols, math_t *components,
              int n_cols_comp, cudaStream_t stream) {

	auto counting = thrust::make_counting_iterator(0);
	auto m = n_rows;

        thrust::for_each(thrust::cuda::par.on(stream),
                        counting, counting + n_cols, [=]__device__(int idx) {
			int d_i = idx * m;
			int end = d_i + m;

			math_t max = 0.0;
			int max_index = 0;
			for (int i = d_i; i < end; i++) {
				math_t val = input[i];
				if (val < 0.0) {
					val = -val;
				}
				if (val > max) {
					max = val;
					max_index = i;
				}
			}

			if (input[max_index] < 0.0) {
				for (int i = d_i; i < end; i++) {
					input[i] = -input[i];
				}

				int len = n_cols * n_cols_comp;
				for (int i = idx; i < len; i = i + n_cols) {
					components[i] = -components[i];
				}
			}
	});

}

/**
 * @brief perform fit operation for the tsvd. Generates eigenvectors, explained vars, singular vals, etc.
 * @input param handle: the internal cuml handle object
 * @input param input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @output param components: the principal components of the input data. Size n_cols * n_components.
 * @output param singular_vals: singular values of the data. Size n_components * 1
 * @input param prms: data structure that includes all the parameters from input size to algorithm.
 */
template<typename math_t>
void tsvdFit(const cumlHandle_impl& handle, math_t *input, math_t *components,
             math_t *singular_vals, paramsTSVD prms) {
    auto stream = handle.getStream();
    auto cublas_handle = handle.getCublasHandle();
    auto allocator = handle.getDeviceAllocator();

    ASSERT(prms.n_cols > 1,
           "Parameter n_cols: number of columns cannot be less than two");
    ASSERT(prms.n_rows > 1,
           "Parameter n_rows: number of rows cannot be less than two");
    ASSERT(prms.n_components > 0,
           "Parameter n_components: number of components cannot be less than one");

    if (prms.n_components > prms.n_cols)
        prms.n_components = prms.n_cols;

    int len = prms.n_cols * prms.n_cols;
    device_buffer<math_t> input_cross_mult(handle.getDeviceAllocator(),
                                           handle.getStream(), len);

    math_t alpha = math_t(1);
    math_t beta = math_t(0);
    LinAlg::gemm(input, prms.n_rows, prms.n_cols, input, input_cross_mult.data(),
                 prms.n_cols, prms.n_cols, CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta,
                 cublas_handle, stream);

    device_buffer<math_t> components_all(handle.getDeviceAllocator(),
                                         handle.getStream(), len);
    device_buffer<math_t> explained_var_all(handle.getDeviceAllocator(),
                                            handle.getStream(), prms.n_cols);

    calEig(handle, input_cross_mult.data(), components_all.data(),
           explained_var_all.data(), prms);

    Matrix::truncZeroOrigin(components_all.data(), prms.n_cols, components,
                            prms.n_components, prms.n_cols, stream);

    math_t scalar = math_t(1);
    Matrix::seqRoot(explained_var_all.data(), singular_vals, scalar, prms.n_components, stream);
}

/**
 * @brief performs fit and transform operations for the tsvd. Generates transformed data, eigenvectors, explained vars, singular vals, etc.
 * @input param handle: the internal cuml handle object
 * @input param input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @output param trans_input: the transformed data. Size n_rows * n_components.
 * @output param components: the principal components of the input data. Size n_cols * n_components.
 * @output param explained_var: explained variances (eigenvalues) of the principal components. Size n_components * 1.
 * @output param explained_var_ratio: the ratio of the explained variance and total variance. Size n_components * 1.
 * @output param singular_vals: singular values of the data. Size n_components * 1
 * @input param prms: data structure that includes all the parameters from input size to algorithm.
 */
template<typename math_t>
void tsvdFitTransform(const cumlHandle_impl& handle, math_t *input,
                      math_t *trans_input, math_t *components,
                      math_t *explained_var, math_t *explained_var_ratio,
                      math_t *singular_vals, paramsTSVD prms) {
    auto stream = handle.getStream();
    auto cublas_handle = handle.getCublasHandle();
    auto cusolver_handle = handle.getcusolverDnHandle();
    auto allocator = handle.getDeviceAllocator();
    ///@todo: make this to be passed via the interface!
    DeviceAllocator mgr = makeDefaultAllocator();

    tsvdFit(handle, input, components, singular_vals, prms);
    tsvdTransform(input, components, trans_input, prms, cublas_handle, stream);

    signFlip(trans_input, prms.n_rows, prms.n_components, components,
             prms.n_cols, stream);

    device_buffer<math_t> mu_trans(allocator, stream, prms.n_components);
    Stats::mean(mu_trans.data(), trans_input, prms.n_components, prms.n_rows, true,
                false, stream);
    Stats::vars(explained_var, trans_input, mu_trans.data(), prms.n_components,
                prms.n_rows, true, false, stream);

    device_buffer<math_t> mu(allocator, stream, prms.n_cols);
    device_buffer<math_t> vars(allocator, stream, prms.n_cols);

    Stats::mean(mu.data(), input, prms.n_cols, prms.n_rows, true, false, stream);
    Stats::vars(vars.data(), input, mu.data(), prms.n_cols, prms.n_rows, true, false, stream);

    device_buffer<math_t> total_vars(allocator, stream, 1);
    Stats::sum(total_vars.data(), vars.data(), 1, prms.n_cols, false, stream);

    math_t total_vars_h;
    updateHostAsync(&total_vars_h, total_vars.data(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    math_t scalar = math_t(1) / total_vars_h;

    LinAlg::scalarMultiply(explained_var_ratio, explained_var, scalar,
                           prms.n_components, stream);
}

/**
 * @brief performs transform operation for the tsvd. Transforms the data to eigenspace.
 * @input param input: the data is transformed. Size n_rows x n_components.
 * @input param components: principal components of the input data. Size n_cols * n_components.
 * @input param prms: data structure that includes all the parameters from input size to algorithm.
 * @input param cublas_handle: cublas handle
 */
template<typename math_t>
void tsvdTransform(math_t *input, math_t *components, math_t *trans_input,
		paramsTSVD prms, cublasHandle_t cublas_handle, cudaStream_t stream) {

	ASSERT(prms.n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than two");
	ASSERT(prms.n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");
	ASSERT(prms.n_components > 0,
			"Parameter n_components: number of components cannot be less than one");

	math_t alpha = math_t(1);
	math_t beta = math_t(0);
	LinAlg::gemm(input, prms.n_rows, prms.n_cols, components, trans_input,
			prms.n_rows, prms.n_components, CUBLAS_OP_N, CUBLAS_OP_T, alpha, beta,
			cublas_handle, stream);
}

/**
 * @brief performs inverse transform operation for the tsvd. Transforms the transformed data back to original data.
 * @input param trans_input: the data is fitted to PCA. Size n_rows x n_components.
 * @input param components: transpose of the principal components of the input data. Size n_components * n_cols.
 * @output param input: the data is fitted to PCA. Size n_rows x n_cols.
 * @input param prms: data structure that includes all the parameters from input size to algorithm.
 * @input param cublas_handle: cublas handle
 */
template<typename math_t>
void tsvdInverseTransform(math_t *trans_input, math_t *components,
		math_t *input, paramsTSVD prms, cublasHandle_t cublas_handle, cudaStream_t stream) {

	ASSERT(prms.n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(prms.n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than one");
	ASSERT(prms.n_components > 0,
			"Parameter n_components: number of components cannot be less than one");

	math_t alpha = math_t(1);
	math_t beta = math_t(0);

	LinAlg::gemm(trans_input, prms.n_rows, prms.n_components, components, input,
			prms.n_rows, prms.n_cols, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta,
                     cublas_handle, stream);

}

/** @} */

}
;
// end namespace ML
