#pragma once

#include "cuda_utils.h"
#include <cub/cub.cuh>


namespace MLCommon {
namespace LinAlg {

/** different types of norms supported on the input buffers */
enum NormType {
    L1Norm = 0,
    L2Norm
};


template <typename Type, int TPB, typename Lambda>
__global__ void normKernel(Type* dots, const Type* data, int D, int N,
                           Lambda op) {
    typedef cub::BlockReduce<Type, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Type thread_data = Type(0);
    int rowStart = blockIdx.x * D;
    for(int i=threadIdx.x;i<D;i+=TPB) {
        int idx = rowStart + i;
        thread_data += op(data[idx]);
    }
    Type acc = BlockReduce(temp_storage).Sum(thread_data);
    if(threadIdx.x == 0) {
        dots[blockIdx.x] = acc;
    }
}

template <typename Type, int TPB>
void normImpl(Type* dots, const Type* data, int D, int N, NormType type) {
    switch(type) {
    case L1Norm:
        normKernel<Type,TPB><<<N,TPB>>>(dots, data, D, N,
                                        [] __device__ (Type in) {
                                            return myAbs(in);
                                        });
        break;
    case L2Norm:
        normKernel<Type,TPB><<<N,TPB>>>(dots, data, D, N,
                                        [] __device__ (Type in) {
                                            return in * in;
                                        });
        break;
    default:
        ASSERT(false, "Invalid norm type passed! [%d]", type);
    };
    CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Compute row-wise norm of the input matrix
 *
 * Row-wise norm is useful while computing pairwise distance matrix, for example.
 * This is used in many clustering algos like knn, kmeans, dbscan, etc... The
 * current implementation is optimized only for bigger values of 'D'.
 *
 * @tparam Type the data type
 * @param dots the output vector of row-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 */
template <typename Type>
void norm(Type* dots, const Type* data, int D, int N, NormType type) {
    if(D <= 32) {
        normImpl<Type, 32>(dots, data, D, N, type);
    } else if(D <= 64) {
        normImpl<Type, 64>(dots, data, D, N, type);
    } else if(D <= 128) {
        normImpl<Type, 128>(dots, data, D, N, type);
    } else {
        normImpl<Type, 256>(dots, data, D, N, type);
    }
}

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
