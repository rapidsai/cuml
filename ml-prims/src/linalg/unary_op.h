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

#include "vectorized.h"
#include "cuda_utils.h"


namespace MLCommon {
namespace LinAlg {

template <typename math_t, int veclen_, typename Lambda>
__global__ void unaryOpKernel(math_t* out, const math_t* in, math_t scalar,
                              int len, Lambda op) {
    typedef TxN_t<math_t,veclen_> VecType;
    VecType a;
    int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * VecType::Ratio;
    if(idx >= len)
        return;
    a.load(in, idx);
    #pragma unroll
    for(int i=0;i<VecType::Ratio;++i) {
        a.val.data[i] = op(a.val.data[i], scalar);
    }
    a.store(out, idx);
}

template <typename math_t, int veclen_, typename Lambda, int TPB>
void unaryOpImpl(math_t* out, const math_t* in, math_t scalar, int len,
                 Lambda op, cudaStream_t stream = 0) {
    const int nblks = ceildiv(veclen_? len/veclen_ : len, TPB);
    unaryOpKernel<math_t,veclen_,Lambda><<<nblks, TPB, 0, stream>>>(out, in, scalar, len, op);
    CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief perform element-wise unary operation in the input array
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in the input array
 * @param scalar scalar used to perform the unary operation
 * @param len number of elements in the input array
 * @param op the device-lambda
 */
template <typename math_t, typename Lambda, int TPB=256>
void unaryOp(math_t* out, const math_t* in, math_t scalar, int len, Lambda op,
		     cudaStream_t stream = 0) {

    size_t bytes = len * sizeof(math_t);
    uint64_t inAddr = uint64_t(in);
    uint64_t outAddr = uint64_t(out);
    if(16/sizeof(math_t) && bytes % 16 == 0 && inAddr % 16 == 0 && outAddr % 16 == 0) {
        unaryOpImpl<math_t,16/sizeof(math_t),Lambda,TPB>(out, in, scalar, len, op, stream);
    } else if(8/sizeof(math_t) && bytes % 8 == 0 && inAddr % 8 == 0 && outAddr % 8 == 0) {
        unaryOpImpl<math_t,8/sizeof(math_t),Lambda,TPB>(out, in, scalar, len, op, stream);
    } else if(4/sizeof(math_t) && bytes % 4 == 0 && inAddr % 4 == 0 && outAddr % 4 == 0) {
        unaryOpImpl<math_t,4/sizeof(math_t),Lambda,TPB>(out, in, scalar, len, op, stream);
    } else if(2/sizeof(math_t) && bytes % 2 == 0 && inAddr % 2 == 0 && outAddr % 2 == 0) {
        unaryOpImpl<math_t,2/sizeof(math_t),Lambda,TPB>(out, in, scalar, len, op, stream);
    } else if(1/sizeof(math_t)) {
        unaryOpImpl<math_t,1/sizeof(math_t),Lambda,TPB>(out, in, scalar, len, op, stream);
    } else {
        unaryOpImpl<math_t,1,Lambda,TPB>(out, in, scalar, len, op, stream);
    }
}

/**
 * @brief perform element-wise unary operation in the input array
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in the input array
 * @param scalar scalar used to perform the unary operation
 * @param len number of elements in the input array
 * @param op the device-lambda
 */
template <typename math_t, typename Lambda, int TPB=256>
void unaryOpMG(TypeMG<math_t>* out, const TypeMG<math_t>* in, math_t scalar, int len,
		     int n_gpus, Lambda op, bool sync = false) {

	for (int i = 0; i < n_gpus; i++) {
		CUDA_CHECK(cudaSetDevice(in[i].gpu_id));

		int len = in[i].n_cols * in[i].n_rows;
		unaryOp(out[i].d_data, in[i].d_data, scalar, len,
					op, in[i].stream);
	}

	if (sync)
		streamSyncMG(in, n_gpus);

}

}; // end namespace LinAlg
}; // end namespace MLCommon
