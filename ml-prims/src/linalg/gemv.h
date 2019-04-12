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

#include "cuda_utils.h"
#include <cublas_v2.h>
#include "cublas_wrappers.h"

namespace MLCommon {
namespace LinAlg {


template<typename math_t>
void gemv(const math_t* a, int n_rows, int n_cols, const math_t* x, int incx,
		math_t* y, int incy, bool trans_a, math_t alpha, math_t beta,
		cublasHandle_t cublas_h, cudaStream_t stream) {

	cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Unfortunately there is a clash of terminology 
    // in BLAS https://docs.nvidia.com/cuda/cublas/index.html is opposite to Machine Learning
    // In blas:
    //  m - number of rows in input matrix
    //  n - number of columns in input matrix
    //  lda - purpose of it  to have ability to operate on submatrices of matrix without copying.
    //        If you're not think about it it's always should be equal to m
    //  lda has deal with memory layout, but has nothing with the requirement for cuBLAS perform transpose

    // In Machine Learning:
    //  m - nunmber of columns in design matrix(number of features)
    //  n - number of rows in designed matrix (number of train examples)

	int m = n_rows;
	int n = n_cols;
	int lda = trans_a ? m : n;

	CUBLAS_CHECK(
			cublasgemv(cublas_h, op_a, m, n, &alpha, a, lda, x, incx,
					&beta, y, incy, stream));
}

template<typename math_t>
void gemv(const math_t* a, int n_rows_a, int n_cols_a, const math_t* x,
		math_t* y, bool trans_a, math_t alpha, math_t beta,
		cublasHandle_t cublas_h, cudaStream_t stream) {

	gemv(a, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, cublas_h, stream);
}

template<typename math_t>
void gemv(const math_t* a, int n_rows_a, int n_cols_a, const math_t* x,
		math_t* y, bool trans_a, cublasHandle_t cublas_h, cudaStream_t stream) {

	math_t alpha = math_t(1);
	math_t beta = math_t(0);

	gemv(a, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, cublas_h, stream);
}

}
;
// end namespace LinAlg
}
;
// end namespace MLCommon
