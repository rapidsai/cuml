#pragma once

#include "cuda_utils.h"
#include "vectorized.h"
#include "linalg/matrix_vector_op.h"

namespace MLCommon {
namespace Stats {

/**
 * @brief Center the input matrix wrt its mean
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam TPB threads per block of the cuda kernel launched
 * @param data matrix which needs to be centered (currently assumed to be row-major)
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 */
template <typename Type, int TPB=256>
void meanCenter(Type* data, const Type* mu, int D, int N, bool rowMajor) {
	LinAlg::matrixVectorOp(data, mu, D, N, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		        		                 return a - b;
		        		            });
}

/**
 * @brief Add the input matrix wrt its mean
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam TPB threads per block of the cuda kernel launched
 * @param data matrix which needs to be centered (currently assumed to be row-major)
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 */
template <typename Type, int TPB=256>
void meanAdd(Type* data, const Type* mu, int D, int N, bool rowMajor) {
	LinAlg::matrixVectorOp(data, mu, D, N, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		        		                 return a + b;
		        		            });
}

}; // end namespace Stats
}; // end namespace MLCommon
