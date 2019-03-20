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
#include "preprocess.h"
#include "common/cumlHandle.hpp"

namespace ML {
namespace GLM {

using namespace MLCommon;

template<typename math_t>
void olsFit(math_t *input, int n_rows, int n_cols, math_t *labels, math_t *coef,
		math_t *intercept, bool fit_intercept, bool normalize,
		cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
		int algo = 0) {
    ///@todo: make this function accept cumlHandle_impl!
    cumlHandle h_;
    cudaStream_t s_;
    CUDA_CHECK(cudaStreamCreate(&s_));
    h_.setStream(s_);
    const auto& handle = h_.getImpl();
    auto allocator = handle.getDeviceAllocator();
    auto stream = handle.getStream();

	ASSERT(n_cols > 0,
			"olsFit: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"olsFit: number of rows cannot be less than two");

        device_buffer<math_t> mu_input(allocator, stream);
        device_buffer<math_t> norm2_input(allocator, stream);
        device_buffer<math_t> mu_labels(allocator, stream);

	if (fit_intercept) {
            mu_input.resize(n_cols, stream);
            mu_labels.resize(1, stream);
            if (normalize) {
                norm2_input.resize(n_cols, stream);
            }
            preProcessData(input, n_rows, n_cols, labels, intercept, mu_input.data(),
                           mu_labels.data(), norm2_input.data(), fit_intercept,
                           normalize, cublas_handle, cusolver_handle);
	}

	if (algo == 0 || n_cols == 1) {
            LinAlg::lstsqSVD(input, n_rows, n_cols, labels, coef, cusolver_handle,
                             cublas_handle, allocator, stream);
	} else if (algo == 1) {
            LinAlg::lstsqEig(input, n_rows, n_cols, labels, coef, cusolver_handle,
                             cublas_handle, allocator, stream);
	} else if (algo == 2) {
		LinAlg::lstsqQR(input, n_rows, n_cols, labels, coef, cusolver_handle,
				cublas_handle);
	} else if (algo == 3) {
		ASSERT(false, "olsFit: no algorithm with this id has been implemented");
	} else {
		ASSERT(false, "olsFit: no algorithm with this id has been implemented");
	}

	if (fit_intercept) {
            postProcessData(input, n_rows, n_cols, labels, coef, intercept, mu_input.data(),
                            mu_labels.data(), norm2_input.data(), fit_intercept, normalize,
                            cublas_handle, cusolver_handle);
	} else {
		*intercept = math_t(0);
	}
    CUDA_CHECK(cudaStreamSynchronize(s_));
    CUDA_CHECK(cudaStreamDestroy(s_));
}

template<typename math_t>
void olsPredict(const math_t *input, int n_rows, int n_cols, const math_t *coef,
		math_t intercept, math_t *preds, cublasHandle_t cublas_handle) {

	ASSERT(n_cols > 0,
			"olsPredict: number of columns cannot be less than one");
	ASSERT(n_rows > 0,
			"olsPredict: number of rows cannot be less than one");

	math_t alpha = math_t(1);
	math_t beta = math_t(0);
	LinAlg::gemm(input, n_rows, n_cols, coef, preds, n_rows, 1, CUBLAS_OP_N,
                     CUBLAS_OP_N, alpha, beta, cublas_handle);

	LinAlg::addScalar(preds, preds, intercept, n_rows);

}

/** @} */
}
;
}
;
// end namespace ML
