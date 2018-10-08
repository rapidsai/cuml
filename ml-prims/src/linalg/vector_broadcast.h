#pragma once

#include "vectorized.h"
#include "cuda_utils.h"


namespace MLCommon {
namespace Broadcast {


///@todo: investigate if using shared mem for vector would help with perf
template <typename math_t, int veclen_, typename Lambda>
__global__ void vectorBcastKernel(math_t* out, const math_t* matrix,
                                  const math_t* vector, int rows, int cols,
                                  Lambda op) {
    typedef TxN_t<math_t,veclen_> VecType;
    VecType mat, vec;
    int len = rows * cols;
    int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * VecType::Ratio;
    int colIdx = idx % cols;
    if(idx >= len)
        return;
    mat.load(matrix, idx);
    vec.load(vector, colIdx);
    #pragma unroll
    for(int i=0;i<VecType::Ratio;++i)
        mat.val.data[i] = op(mat.val.data[i], vec.val.data[i]);
    mat.store(out, idx);
}

template <typename math_t, int veclen_, typename Lambda, int TPB>
void vectorBcastImpl(math_t* out, const math_t* matrix, const math_t* vector,
                     int rows, int cols, Lambda op) {
    int len = rows * cols;
    const int nblks = ceildiv(veclen_? len/veclen_ : len, TPB);
    vectorBcastKernel<math_t,veclen_,Lambda><<<nblks,TPB>>>
        (out, matrix, vector, rows, cols, op);
    CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief perform element-wise binary operation between a matrix and a vector,
 * with the vector being broadcasted across the other dimension of the matrix.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output matrix
 * @param matrix the input matrix (dimension = rows x cols)
 * @param vector the input vector (length = cols)
 * @param rows number of rows in the input/output matrix
 * @param cols number of cols in the input/output matrix
 * @param op the device-lambda
 * @note If you want to work on a column-major storage with each column
 * wanting to work on the input vector, then just swap rows and cols while
 * calling this method.
 */
template <typename math_t, typename Lambda, int TPB=128>
void vectorBroadcast(math_t* out, const math_t* matrix, const math_t* vector,
                     int rows, int cols, Lambda op) {
    // need to use 'cols' here since vector access is based on this!
    size_t bytes = cols * sizeof(math_t);
    if(16/sizeof(math_t) && bytes % 16 == 0) {
        vectorBcastImpl<math_t,16/sizeof(math_t),Lambda,TPB>
            (out, matrix, vector, rows, cols, op);
    } else if(8/sizeof(math_t) && bytes % 8 == 0) {
        vectorBcastImpl<math_t,8/sizeof(math_t),Lambda,TPB>
            (out, matrix, vector, rows, cols, op);
    } else if(4/sizeof(math_t) && bytes % 4 == 0) {
        vectorBcastImpl<math_t,4/sizeof(math_t),Lambda,TPB>
            (out, matrix, vector, rows, cols, op);
    } else if(2/sizeof(math_t) && bytes % 2 == 0) {
        vectorBcastImpl<math_t,2/sizeof(math_t),Lambda,TPB>
            (out, matrix, vector, rows, cols, op);
    } else if(1/sizeof(math_t)) {
        vectorBcastImpl<math_t,1/sizeof(math_t),Lambda,TPB>
            (out, matrix, vector, rows, cols, op);
    } else {
        vectorBcastImpl<math_t,1,Lambda,TPB>
            (out, matrix, vector, rows, cols, op);
    }
}

}; // end namespace Broadcast
}; // end namespace MLCommon
