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

#include "linalg/coalesced_reduction.h"


namespace MLCommon {
namespace LinAlg {

/** different types of norms supported on the input buffers */
enum NormType { L1Norm = 0, L2Norm };


/**
 * @brief Compute row-wise norm of the input matrix
 *
 * Row-wise norm is useful while computing pairwise distance matrix, for
 * example.
 * This is used in many clustering algos like knn, kmeans, dbscan, etc... The
 * current implementation is optimized only for bigger values of 'D'.
 *
 * @tparam Type the data type
 * @param dots the output vector of row-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param fin_op the final lambda op
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda>
void norm(Type *dots, const Type *data, int D, int N, NormType type,
          Lambda fin_op, cudaStream_t stream = 0) {
  switch (type) {
    case L1Norm:
      coalescedReduction(dots, data, D, N, (Type)0,
                         false, stream,
                         [] __device__(Type in) { return myAbs(in); }, 
                         [] __device__(Type a, Type b) { return a+b; }, fin_op);
      break;
    case L2Norm:
      coalescedReduction(dots, data, D, N, (Type)0,
                         false, stream,
                         [] __device__(Type in) { return in * in; },
                         [] __device__(Type a, Type b) { return a+b; }, fin_op);
      break;
    default:
      ASSERT(false, "Invalid norm type passed! [%d]", type);
  };
}

///@todo: we need to remove this and use the above methods!!!!
template <typename Type, int TPB>
__global__ void norm2KernelColMajor(Type* norm2, const Type* data, int D, int N) {
    typedef cub::BlockReduce<Type, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Type thread_data = Type(0);
    int colStart = blockIdx.x * N;
    for(int i=threadIdx.x;i<N;i+=TPB) {
        int idx = colStart + i;
        thread_data += data[idx] * data[idx];
    }
    Type acc = BlockReduce(temp_storage).Sum(thread_data);
    if(threadIdx.x == 0) {
    	norm2[blockIdx.x] = MLCommon::mySqrt(acc);
    }
}

/**
 * @brief Compute row-wise norm of the input matrix without the fin_op lambda
 * @tparam Type the data type
 * @param dots the output vector of row-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param stream cuda stream where to launch work
 */
template <typename Type>
void norm(Type *dots, const Type *data, int D, int N, NormType type,
          cudaStream_t stream = 0) {
  norm(dots, data, D, N, type, [] __device__(Type in) { return in; }, stream);
}

/**
 * @brief Compute norm2 of the input matrix
 *
 * Column-wise norm is useful to normalize the data for many ML algorithms.
 *
 * @tparam Type the data type
 * @param nrm2 the output vector of row-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 */
template <typename Type>
void norm2(Type* out, const Type* data, int D, int N, bool rowMajor=false) {
	if (rowMajor)
		ASSERT(true, "norm.h: row major norm is not implemented. This parameter is for future use only.");

	static const int TPB = 256;
	norm2KernelColMajor<Type,TPB><<<D,TPB>>>(out, data, D, N);
	CUDA_CHECK(cudaPeekAtLastError());
}

}; // end namespace LinAlg
}; // end namespace MLCommon
