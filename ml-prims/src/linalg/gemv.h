#pragma once

#include "cuda_utils.h"
#include <cublas_v2.h>
#include "cublas_wrappers.h"

namespace MLCommon {
namespace LinAlg {


template<typename math_t>
void gemv(const math_t* a, int n_rows, int n_cols, const math_t* x, int incx,
		math_t* y, int incy, bool trans_a, math_t alpha, math_t beta,
		cublasHandle_t cublas_h) {

	cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

	int m = n_rows;
	int n = n_cols;
	int lda = trans_a ? m : n;

	CUBLAS_CHECK(
			cublasgemv(cublas_h, op_a, m, n, &alpha, a, lda, x, incx,
					&beta, y, incy));
}

template<typename math_t>
void gemv(const math_t* a, int n_rows_a, int n_cols_a, const math_t* x,
		math_t* y, bool trans_a, math_t alpha, math_t beta,
		cublasHandle_t cublas_h) {

	gemv(a, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, cublas_h);
}

template<typename math_t>
void gemv(const math_t* a, int n_rows_a, int n_cols_a, const math_t* x,
		math_t* y, bool trans_a, cublasHandle_t cublas_h) {

	math_t alpha = math_t(1);
	math_t beta = math_t(0);

	gemv(a, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, cublas_h);
}

}
;
// end namespace LinAlg
}
;
// end namespace MLCommon
