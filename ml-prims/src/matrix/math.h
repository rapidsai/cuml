#pragma once

#include "linalg/unary_op.h"
#include "linalg/binary_op.h"
#include <thrust/inner_product.h>
#include "linalg/matrix_vector_op.h"

namespace MLCommon {
namespace Matrix {

using namespace MLCommon::LinAlg;

/**
 * @defgroup power math operation on the input matrix. Power of every element in the input matrix
 * @param inout: input matrix and also the result is stored
 * @param scalar: every element is multiplied with scalar.
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void power(math_t* inout, math_t scalar, int len) {

	auto counting = thrust::make_counting_iterator(0);
	auto d_A = inout;

    thrust::for_each(counting, counting + len, [=]__device__(int idx)
	{
		d_A[idx] = d_A[idx] * d_A[idx] * scalar;
	});
}

/**
 * @defgroup power math operation on the input matrix. Power of every element in the input matrix
 * @param in: input matrix
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar.
 * @param len: number elements of input matrix
 * @{
 */
template<typename math_t>
void power(math_t* in, math_t* out, math_t scalar, int len) {

	auto counting = thrust::make_counting_iterator(0);
	auto d_src = in;
	auto d_dest = out;

    thrust::for_each(counting, counting + len, [=]__device__(int idx)
	{
    	d_dest[idx] = d_src[idx] * d_src[idx] * scalar;
	});
}

/**
 * @defgroup overloaded power math operation on the input matrix. Power of every element in the input matrix
 * @param inout: input matrix and also the result is stored
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void power(math_t* inout, int len) {
	math_t scalar = 1.0;
	power(inout, scalar, len);
}

/**
 * @defgroup power math operation on the input matrix. Power of every element in the input matrix
 * @param in: input matrix
 * @param out: output matrix. The result is stored in the out matrix
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void power(math_t* in, math_t* out, int len) {
	math_t scalar = 1.0;
	power(in, out, scalar, len);
}

/**
 * @defgroup square root math operation on the input matrix. Square root of every element in the input matrix
 * @param inout: input matrix and also the result is stored
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void seqRoot(math_t* inout, math_t scalar, int len) {

	auto counting = thrust::make_counting_iterator(0);
	auto d_A = inout;

    thrust::for_each(counting, counting + len, [=]__device__(int idx)
	{
    	d_A[idx] = sqrt(d_A[idx] * scalar);
	});
}

/**
 * @defgroup square root math operation on the input matrix. Square root of every element in the input matrix
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @{
 */
template<typename math_t>
void seqRoot(math_t* in, math_t* out, math_t scalar, int len) {

	auto counting = thrust::make_counting_iterator(0);
	auto d_src = in;
	auto d_dest = out;

    thrust::for_each(counting, counting + len, [=]__device__(int idx)
	{
    	d_dest[idx] = sqrt(d_src[idx] * scalar);
	});
}

/**
 * @defgroup overloaded square root math operation on the input matrix. Square root of every element in the input matrix
 * @param inout: input matrix and also the result is stored
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void seqRoot(math_t* inout, int len) {
	math_t scalar = 1.0;
	seqRoot(inout, scalar, len);
}

/**
 * @defgroup square root math operation on the input matrix. Square root of every element in the input matrix
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void seqRoot(math_t* in, math_t* out, int len) {
	math_t scalar = 1.0;
	seqRoot(in, out, scalar, len);
}

/**
 * @defgroup inverse math operation on the input matrix. Reciprocal of every element in the input matrix
 * @param inout: input matrix and also the result is stored
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void reciprocal(math_t* inout, math_t scalar, int len) {

	auto counting = thrust::make_counting_iterator(0);
	auto d_A = inout;

    thrust::for_each(counting, counting + len, [=]__device__(int idx)
	{
    	d_A[idx] = scalar / d_A[idx];
	});
}

/**
 * @defgroup inverse math operation on the input matrix. Reciprocal of every element in the input matrix
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @{
 */
template<typename math_t>
void reciprocal(math_t* in, math_t* out, math_t scalar, int len) {

	auto counting = thrust::make_counting_iterator(0);
	auto d_src = in;
	auto d_dest = out;

    thrust::for_each(counting, counting + len, [=]__device__(int idx)
	{
    	d_dest[idx] = scalar / d_src[idx];
	});
}

/**
 * @defgroup overloaded reciprocal math operation on the input matrix. Reciprocal of every element in the input matrix
 * @param inout: input matrix and also the result is stored
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void reciprocal(math_t* inout, int len) {
	math_t scalar = 1.0;
	reciprocal(inout, scalar, len);
}

/**
 * @defgroup inverse math operation on the input matrix. Reciprocal of every element in the input matrix
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void reciprocal(math_t* in, math_t* out, int len) {
	math_t scalar = 1.0;
	reciprocal(in, out, scalar, len);
}

/**
 * @defgroup ratio math operation on the input matrix. ratio of every element over sum of input vector is calculated
 *           Used in PCA.
 * @param src: input matrix
 * @param dest: output matrix. The result is stored in the dest matrix
 * @param len: number elements of input matrix
 * @{
 */

// TODO: Check with Thejaswi if he can come up with faster approach to this function.
template <typename math_t>
void ratio(math_t *src, math_t *dest, int len) {

	auto counting = thrust::make_counting_iterator(0);
	auto d_src = src;
	auto s = len;
	auto d_dest = dest;

    thrust::for_each(counting, counting + len, [=]__device__(int idx)
	{
    	math_t total = 0.0;
    	for (int i = 0; i < s; i++) {
    		total += d_src[i];
    	}
    	if (total != 0.0) {
    	    d_dest[idx] = d_src[idx] / total;
    	}
	});
}

/**
 * @defgroup sign flip for PCA. This is used to stabilize the sign of column major eigen vectors
 * @param inout: input matrix. Result also stored in this parameter
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @{
 */
// TODO: Check with Thejaswi if he can come up with faster approach to this function.
template<typename math_t>
void signFlip(math_t *inout, int n_rows, int n_cols) {

	auto counting = thrust::make_counting_iterator(0);
	auto m = n_rows;

	thrust::for_each(counting, counting + n_cols, [=]__device__(int idx) {
		int d_i = idx * m;
		int end = d_i + m;

		math_t max = 0.0;
	    int max_index = 0;
		for (int i = d_i; i < end; i++) {
			math_t val = inout[i];
			if (val < 0.0) {
				val = -val;
			}
			if (val > max) {
				max = val;
				max_index = i;
			}
		}

		if (inout[max_index] < 0.0) {
			for (int i = d_i; i < end; i++) {
				inout[i] = -inout[i];
			}
		}
	});

}

template <typename Type, int TPB=256>
void matrixVectorBinaryMult(Type* data, const Type* vec, int n_row, int n_col, bool rowMajor) {
	matrixVectorOp(data, vec, n_col, n_row, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		        		                 return a * b;
		        		            });
}

template <typename Type, int TPB=256>
void matrixVectorBinaryMultSkipZero(Type* data, const Type* vec, int n_row, int n_col, bool rowMajor) {
	matrixVectorOp(data, vec, n_col, n_row, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		                              if (b == Type(0))
				                         return a;
		                              else
		        		                 return a * b;
		        		            });
}

template <typename Type, int TPB=256>
void matrixVectorBinaryDiv(Type* data, const Type* vec, int n_row, int n_col, bool rowMajor) {
	matrixVectorOp(data, vec, n_col, n_row, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		        		                 return a / b;
		        		            });
}

template <typename Type, int TPB=256>
void matrixVectorBinaryDivSkipZero(Type* data, const Type* vec, int n_row, int n_col, bool rowMajor) {
	matrixVectorOp(data, vec, n_col, n_row, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		                               if (b == Type(0))
		                            	   return a;
		                               else
		        		                   return a / b;
		        		            });
}

template <typename Type, int TPB=256>
void matrixVectorBinaryAdd(Type* data, const Type* vec, int n_row, int n_col, bool rowMajor) {
	matrixVectorOp(data, vec, n_col, n_row, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		        		                 return a + b;
		        		            });
}

template <typename Type, int TPB=256>
void matrixVectorBinarySub(Type* data, const Type* vec, int n_row, int n_col, bool rowMajor) {
	matrixVectorOp(data, vec, n_col, n_row, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		        		                 return a - b;
		        		            });
}

}; // end namespace Matrix
}; // end namespace MLCommon

