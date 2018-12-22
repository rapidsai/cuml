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
#include "linalg/matrix_vector_op.h"

namespace MLCommon {
namespace Stats {

using namespace MLCommon;

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

template <typename Type, int TPB=256>
void meanCenterMG(TypeMG<Type>* data, TypeMG<Type>* mu, int D, int N, int n_gpus, bool rowMajor,
		bool row_split = false, bool sync = false) {

	LinAlg::matrixVectorOpMG(data, mu, D, N, n_gpus, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		        		                 return a - b;
		        		            }, row_split, sync);
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

template <typename Type, int TPB=256>
void meanAddMG(TypeMG<Type>* data, TypeMG<Type>* mu, int D, int N, int n_gpus, bool rowMajor,
		bool row_split = false, bool sync = false) {

	LinAlg::matrixVectorOpMG(data, mu, D, N, n_gpus, rowMajor,
		        		       [] __device__ (Type a, Type b) {
		        		                 return a + b;
		        		            }, row_split, sync);
}

}; // end namespace Stats
}; // end namespace MLCommon
