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
#include <linalg/lstsq.h>
#include <linalg/gemv.h>
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <stats/stddev.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/norm.h>
#include <stats/sum.h>
#include <matrix/math.h>
/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <matrix/matrix.h>
#include "preprocess.h"
#include <device_allocator.h>

namespace ML {
namespace GLM {

using namespace MLCommon;

template<typename math_t>
void olsFit(math_t *input, int n_rows, int n_cols, math_t *labels, math_t *coef,
		math_t *intercept, bool fit_intercept, bool normalize,
		cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
		cudaStream_t stream, int algo = 0) {

	ASSERT(n_cols > 0,
			"olsFit: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"olsFit: number of rows cannot be less than two");

	math_t *mu_input, *norm2_input, *mu_labels;

	if (fit_intercept) {
		allocate(mu_input, n_cols);
		allocate(mu_labels, 1);
		if (normalize) {
			allocate(norm2_input, n_cols);
		}
		preProcessData(input, n_rows, n_cols, labels, intercept, mu_input,
				mu_labels, norm2_input, fit_intercept, normalize, cublas_handle,
				cusolver_handle, stream);
	}

        ///@todo: for perf reasons we should be using custom allocators!
        DeviceAllocator mgr = makeDefaultAllocator();
	if (algo == 0 || n_cols == 1) {
		LinAlg::lstsqSVD(input, n_rows, n_cols, labels, coef, cusolver_handle,
                                 cublas_handle, mgr, stream);
	} else if (algo == 1) {
		LinAlg::lstsqEig(input, n_rows, n_cols, labels, coef, cusolver_handle,
                                 cublas_handle, mgr, stream);
	} else if (algo == 2) {
		LinAlg::lstsqQR(input, n_rows, n_cols, labels, coef, cusolver_handle,
				cublas_handle, stream);
	} else if (algo == 3) {
		ASSERT(false, "olsFit: no algorithm with this id has been implemented");
	} else {
		ASSERT(false, "olsFit: no algorithm with this id has been implemented");
	}

	if (fit_intercept) {
		postProcessData(input, n_rows, n_cols, labels, coef, intercept, mu_input,
				mu_labels, norm2_input, fit_intercept, normalize, cublas_handle,
				cusolver_handle, stream);

		if (normalize) {
			if (norm2_input != NULL)
				cudaFree(norm2_input);
		}

		if (mu_input != NULL)
			cudaFree(mu_input);
		if (mu_labels != NULL)
			cudaFree(mu_labels);
	} else {
		*intercept = math_t(0);
	}

}

template<typename math_t>
void olsPredict(const math_t *input, int n_rows, int n_cols, const math_t *coef,
		math_t intercept, math_t *preds, cublasHandle_t cublas_handle, cudaStream_t stream) {

	ASSERT(n_cols > 0,
			"olsPredict: number of columns cannot be less than one");
	ASSERT(n_rows > 0,
			"olsPredict: number of rows cannot be less than one");

	math_t alpha = math_t(1);
	math_t beta = math_t(0);
	LinAlg::gemm(input, n_rows, n_cols, coef, preds, n_rows, 1, CUBLAS_OP_N,
                     CUBLAS_OP_N, alpha, beta, cublas_handle, stream);

	LinAlg::addScalar(preds, preds, intercept, n_rows, stream);

}

/** @} */
}
;
}
;
// end namespace ML
