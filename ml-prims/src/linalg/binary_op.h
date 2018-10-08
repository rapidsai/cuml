#pragma once

#include "vectorized.h"
#include "cuda_utils.h"


namespace MLCommon {
namespace LinAlg {

template <typename math_t, int veclen_, typename Lambda>
__global__ void binaryOpKernel(math_t* out, const math_t* in1, const math_t* in2,
                               int len, Lambda op) {
    typedef TxN_t<math_t,veclen_> VecType;
    VecType a, b;
    int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * VecType::Ratio;
    if(idx >= len)
        return;
    a.load(in1, idx);
    b.load(in2, idx);
    #pragma unroll
    for(int i=0;i<VecType::Ratio;++i) {
        a.val.data[i] = op(a.val.data[i], b.val.data[i]);
    }
    a.store(out, idx);
}

template <typename math_t, int veclen_, typename Lambda, int TPB>
void binaryOpImpl(math_t* out, const math_t* in1, const math_t* in2,
                  int len, Lambda op) {
    const int nblks = ceildiv(veclen_? len/veclen_ : len, TPB);
    binaryOpKernel<math_t,veclen_,Lambda><<<nblks,TPB>>>(out, in1, in2, len, op);
    CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief perform element-wise unary operation in the input array
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in1 the first input array
 * @param in2 the second input array
 * @param len number of elements in the input array
 * @param op the device-lambda
 */
template <typename math_t, typename Lambda, int TPB=256>
void binaryOp(math_t* out, const math_t* in1, const math_t* in2,
              int len, Lambda op) {
    size_t bytes = len * sizeof(math_t);
    if(16/sizeof(math_t) && bytes % 16 == 0) {
        binaryOpImpl<math_t,16/sizeof(math_t),Lambda,TPB>(out, in1, in2, len, op);
    } else if(8/sizeof(math_t) && bytes % 8 == 0) {
        binaryOpImpl<math_t,8/sizeof(math_t),Lambda,TPB>(out, in1, in2, len, op);
    } else if(4/sizeof(math_t) && bytes % 4 == 0) {
        binaryOpImpl<math_t,4/sizeof(math_t),Lambda,TPB>(out, in1, in2, len, op);
    } else if(2/sizeof(math_t) && bytes % 2 == 0) {
        binaryOpImpl<math_t,2/sizeof(math_t),Lambda,TPB>(out, in1, in2, len, op);
    } else if(1/sizeof(math_t)) {
        binaryOpImpl<math_t,1/sizeof(math_t),Lambda,TPB>(out, in1, in2, len, op);
    } else {
        binaryOpImpl<math_t,1,Lambda,TPB>(out, in1, in2, len, op);
    }
}

}; // end namespace LinAlg
}; // end namespace MLCommon
