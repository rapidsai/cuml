#pragma once

#include "cuda_utils.h"
#include "vectorized.h"


namespace MLCommon {
namespace LinAlg {


template <typename Type, int veclen_, typename Lambda, int TPB>
__global__ void matrixVectorOpKernelRowMajor(Type* matrix, const Type* vec, int D,
                                         int N, Lambda op) {
    typedef TxN_t<Type,veclen_> VecType;
    VecType a, b;
    int rowStart = blockIdx.x * D;
    const int stride = TPB * VecType::Ratio;
    for(int i=threadIdx.x*VecType::Ratio;i<D;i+=stride) {
        a.load(matrix, i+rowStart);
        b.load(vec, i);
        #pragma unroll
        for(int j=0;j<VecType::Ratio;++j)
            a.val.data[j] = op(a.val.data[j], b.val.data[j]);
        a.store(matrix, i+rowStart);
    }
}

template <typename Type, int veclen_, typename Lambda, int TPB>
__global__ void matrixVectorOpKernelColMajor(Type* matrix, const Type* vec, int D,
                                         int N, Lambda op) {
    typedef TxN_t<Type,veclen_> VecType;
    VecType a;
    Type b = vec[blockIdx.x];
    int colStart = blockIdx.x * N;
    const int stride = TPB * VecType::Ratio;
    for(int i=threadIdx.x*VecType::Ratio;i<N;i+=stride) {
        a.load(matrix, i+colStart);
        #pragma unroll
        for(int j=0;j<VecType::Ratio;++j)
            a.val.data[j] = op( a.val.data[j], b);
        a.store(matrix, i+colStart);
    }
}

template <typename Type, int veclen_, typename Lambda, int TPB>
void matrixVectorOpImpl(Type* matrix, const Type* vec, int D, int N, bool rowMajor, Lambda op) {
    if(rowMajor) {
    	matrixVectorOpKernelRowMajor<Type,veclen_,Lambda,TPB><<<N,TPB>>>(matrix, vec, D, N, op);
    } else {
    	matrixVectorOpKernelColMajor<Type,veclen_,Lambda,TPB><<<D,TPB>>>(matrix, vec, D, N, op);
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Center the input matrix wrt its mean
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type the matrix type
 * @tparam TPB threads per block of the cuda kernel launched
 * @param matrix matrix which needs to be centered (currently assumed to be row-major)
 * @param vec the mean vector
 * @param D number of columns of matrix
 * @param N number of rows of matrix
 * @param rowMajor whether input is row or col major
 */
template <typename Type, typename Lambda, int TPB=256>
void matrixVectorOp(Type* matrix, const Type* vec, int D, int N, bool rowMajor, Lambda op) {
    int stride = rowMajor? D : N;
    size_t bytes = stride * sizeof(Type);
    if(16/sizeof(Type) && bytes % 16 == 0) {
    	matrixVectorOpImpl<Type,16/sizeof(Type),Lambda,TPB>(matrix, vec, D, N, rowMajor, op);
    } else if(8/sizeof(Type) && bytes % 8 == 0) {
    	matrixVectorOpImpl<Type,8/sizeof(Type),Lambda,TPB>(matrix, vec, D, N, rowMajor, op);
    } else if(4/sizeof(Type) && bytes % 4 == 0) {
    	matrixVectorOpImpl<Type,4/sizeof(Type),Lambda,TPB>(matrix, vec, D, N, rowMajor, op);
    } else if(2/sizeof(Type) && bytes % 2 == 0) {
    	matrixVectorOpImpl<Type,2/sizeof(Type),Lambda,TPB>(matrix, vec, D, N, rowMajor, op);
    } else if(1/sizeof(Type)) {
    	matrixVectorOpImpl<Type,1/sizeof(Type),Lambda,TPB>(matrix, vec, D, N, rowMajor, op);
    } else {
    	matrixVectorOpImpl<Type,1,Lambda,TPB>(matrix, vec, D, N, rowMajor, op);
    }
}

}; // end namespace Stats
}; // end namespace MLCommon
