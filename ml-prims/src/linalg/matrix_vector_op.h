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
#include "vectorized.h"

namespace MLCommon {
namespace LinAlg {

using namespace MLCommon;

template<typename Type, int veclen_, typename Lambda, int TPB>
__global__ void matrixVectorOpKernelRowMajor(Type* matrix, const Type* vec,
		int D, int N, Lambda op) {
	typedef TxN_t<Type, veclen_> VecType;
	VecType a, b;
	int rowStart = blockIdx.x * D;
	const int stride = TPB * VecType::Ratio;
	for (int i = threadIdx.x * VecType::Ratio; i < D; i += stride) {
		a.load(matrix, i + rowStart);
		b.load(vec, i);
#pragma unroll
		for (int j = 0; j < VecType::Ratio; ++j)
			a.val.data[j] = op(a.val.data[j], b.val.data[j]);
		a.store(matrix, i + rowStart);
	}
}

template<typename Type, int veclen_, typename Lambda, int TPB>
__global__ void matrixVectorOpKernelColMajor(Type* matrix, const Type* vec,
		int D, int N, Lambda op) {
	typedef TxN_t<Type, veclen_> VecType;
	VecType a;
	Type b = vec[blockIdx.x];
	int colStart = blockIdx.x * N;
	const int stride = TPB * VecType::Ratio;
	for (int i = threadIdx.x * VecType::Ratio; i < N; i += stride) {
		a.load(matrix, i + colStart);
#pragma unroll
		for (int j = 0; j < VecType::Ratio; ++j)
			a.val.data[j] = op(a.val.data[j], b);
		a.store(matrix, i + colStart);
	}
}

template<typename Type, int veclen_, typename Lambda, int TPB>
void matrixVectorOpImpl(Type* matrix, const Type* vec, int D, int N,
		bool rowMajor, Lambda op, cudaStream_t stream = 0) {
	if (rowMajor) {
		matrixVectorOpKernelRowMajor<Type, veclen_, Lambda, TPB> <<<N, TPB, 0,
				stream>>>(matrix, vec, D, N, op);
	} else {
		matrixVectorOpKernelColMajor<Type, veclen_, Lambda, TPB> <<<D, TPB, 0,
				stream>>>(matrix, vec, D, N, op);
	}
	CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Operations for all the columns or rows with a given vector.
 *
 * Operations for all the columns or rows with a given vector.
 *
 * @tparam Type: the matrix type
 * @tparam TPB: threads per block of the cuda kernel launched
 * @param matrix: matrix which needs to be centered (currently assumed to be row-major)
 * @param vec: the mean vector
 * @param D: number of columns of matrix
 * @param N: number of rows of matrix
 * @param rowMajor: whether input is row or col major
 * @param op:the mathematical operations
 */
template<typename Type, typename Lambda, int TPB = 256>
void matrixVectorOp(Type* matrix, const Type* vec, int D, int N, bool rowMajor,
		Lambda op, cudaStream_t stream = 0) {
	int stride = rowMajor ? D : N;
	size_t bytes = stride * sizeof(Type);
	if (16 / sizeof(Type) && bytes % 16 == 0) {
		matrixVectorOpImpl<Type, 16 / sizeof(Type), Lambda, TPB>(matrix, vec, D,
				N, rowMajor, op, stream);
	} else if (8 / sizeof(Type) && bytes % 8 == 0) {
		matrixVectorOpImpl<Type, 8 / sizeof(Type), Lambda, TPB>(matrix, vec, D,
				N, rowMajor, op, stream);
	} else if (4 / sizeof(Type) && bytes % 4 == 0) {
		matrixVectorOpImpl<Type, 4 / sizeof(Type), Lambda, TPB>(matrix, vec, D,
				N, rowMajor, op, stream);
	} else if (2 / sizeof(Type) && bytes % 2 == 0) {
		matrixVectorOpImpl<Type, 2 / sizeof(Type), Lambda, TPB>(matrix, vec, D,
				N, rowMajor, op, stream);
	} else if (1 / sizeof(Type)) {
		matrixVectorOpImpl<Type, 1 / sizeof(Type), Lambda, TPB>(matrix, vec, D,
				N, rowMajor, op, stream);
	} else {
		matrixVectorOpImpl<Type, 1, Lambda, TPB>(matrix, vec, D, N, rowMajor,
				op, stream);
	}
}



/**
 * @brief Multi-GPU operations for all the columns or rows with a given vector.
 *
 * Multi-GPU operations for all the columns or rows with a given vector.
 *
 * @tparam Type: the matrix type
 * @tparam TPB: threads per block of the cuda kernel launched
 * @param matrix: matrix which needs to be centered (currently assumed to be row-major)
 * @param vec: the mean vector
 * @param D: number of columns of matrix
 * @param N: number of rows of matrix
 * @param rowMajor: whether input is row or col major
 * @param op:the mathematical operations
 * @param row_split: true if the data is broken by row
 * @param sync: synch the streams if it's true
 */
template<typename Type, typename Lambda>
void matrixVectorOpMG(TypeMG<Type>* matrix, const TypeMG<Type>* vec, int D,
		             int N, int n_gpus, bool rowMajor, Lambda op, bool row_split = false,
		             bool sync = false) {

	if (row_split) {
		ASSERT(false, "matrixVectorOpMG: row split is not supported");
	} else {
		for (int i = 0; i < n_gpus; i++) {
			CUDA_CHECK(cudaSetDevice(matrix[i].gpu_id));

			matrixVectorOp(matrix[i].d_data, vec[i].d_data, matrix[i].n_cols,
					matrix[i].n_rows, rowMajor, op, matrix[i].stream);
		}
	}

	if (sync)
		streamSyncMG(matrix, n_gpus);
}

}
;
// end namespace Stats
}
;
// end namespace MLCommon
