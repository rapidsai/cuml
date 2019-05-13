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
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <stats/stddev.h>
#include <linalg/norm.h>
#include <linalg/gemm.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include "common/cumlHandle.hpp"

namespace ML {
namespace GLM {

using namespace MLCommon;

template<typename math_t>
void preProcessData(const cumlHandle_impl& handle, math_t *input, int n_rows,
		int n_cols, math_t *labels, math_t *intercept, math_t *mu_input,
		math_t *mu_labels, math_t *norm2_input, bool fit_intercept,
		bool normalize, cudaStream_t stream) {

	auto cublas_handle = handle.getCublasHandle();
	auto cusolver_handle = handle.getcusolverDnHandle();

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

	if (fit_intercept) {
		Stats::mean(mu_input, input, n_cols, n_rows, false, false, stream);
		Stats::meanCenter(input, input, mu_input, n_cols, n_rows, false, true,
				stream);

		Stats::mean(mu_labels, labels, 1, n_rows, false, false, stream);
		Stats::meanCenter(labels, labels, mu_labels, 1, n_rows, false, true,
				stream);

		if (normalize) {
			LinAlg::colNorm(norm2_input, input, n_cols, n_rows, LinAlg::L2Norm, false,
					stream,
					[]__device__(math_t v) {return MLCommon::mySqrt(v);});
			Matrix::matrixVectorBinaryDivSkipZero(input, norm2_input, n_rows,
					n_cols, false, true, stream, true);
		}
	}

}

template<typename math_t>
void postProcessData(const cumlHandle_impl& handle, math_t *input, int n_rows,
		int n_cols, math_t *labels, math_t *coef, math_t *intercept,
		math_t *mu_input, math_t *mu_labels, math_t *norm2_input,
		bool fit_intercept, bool normalize, cudaStream_t stream) {

	auto cublas_handle = handle.getCublasHandle();
	auto cusolver_handle = handle.getcusolverDnHandle();

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

	auto allocator = handle.getDeviceAllocator();
	device_buffer<math_t> d_intercept(allocator, stream, 1);

	if (normalize) {
		Matrix::matrixVectorBinaryMult(input, norm2_input, n_rows, n_cols,
				false, true, stream);
		Matrix::matrixVectorBinaryDivSkipZero(coef, norm2_input, 1, n_cols,
				false, true, stream, true);
	}

	LinAlg::gemm(mu_input, 1, n_cols, coef, d_intercept.data(), 1, 1,
			CUBLAS_OP_N, CUBLAS_OP_N, cublas_handle, stream);

	LinAlg::subtract(d_intercept.data(), mu_labels, d_intercept.data(), 1,
			stream);
	updateHost(intercept, d_intercept.data(), 1, stream);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	Stats::meanAdd(input, input, mu_input, n_cols, n_rows, false, true, stream);
	Stats::meanAdd(labels, labels, mu_labels, 1, n_rows, false, true, stream);

}

/** @} */
}
;
}
;
// end namespace ML
