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

#include "binary_op.h"
#include "unary_op.h"
#include "cuda_utils.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in the input buffer
 * @param scalar the scalar used in the operations
 * @param len number of elements in the input buffer
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename math_t, typename IdxType = int>
void subtractScalar(math_t *out, const math_t *in, math_t scalar, IdxType len,
                    cudaStream_t stream) {
  unaryOp(out, in, len,
          [scalar] __device__(math_t in) { return in - scalar; },
          stream);
}

/**
 * @defgroup BinaryOps Element-wise binary operations on the input buffers
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in1 the first input buffer
 * @param in2 the second input buffer
 * @param len number of elements in the input buffers
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename math_t, typename IdxType = int>
void subtract(math_t *out, const math_t *in1, const math_t *in2, IdxType len,
              cudaStream_t stream) {
  binaryOp(out, in1, in2, len,
           [] __device__(math_t a, math_t b) { return a - b; }, stream);
}


template<class math_t, typename IdxType>
__global__ void subtract_dev_scalar_kernel(math_t* outDev, const math_t* inDev,
                                           const math_t *singleScalarDev, IdxType len) {
    //TODO: kernel do not use shared memory in current implementation
    int i = ((IdxType)blockIdx.x * (IdxType)blockDim.x) + threadIdx.x;
    if (i < len) {
        outDev[i] = inDev[i] - *singleScalarDev;
    }
}

/** Substract single value pointed by singleScalarDev parameter in device memory from inDev[i] and write result to outDev[i]
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in the input buffer
 * @param singleScalarDev pointer to the scalar located in device memory
 * @param len number of elements in the input and output buffer
 * @remark block size has not been tuned
 */
template <typename math_t, typename IdxType = int, int TPB = 256>
void subtractDevScalar(math_t* outDev, const math_t* inDev, const math_t* singleScalarDev,
                       IdxType len, cudaStream_t stream)
{
    // Just for the note - there is no way to express such operation with cuBLAS in effective way
    // https://stackoverflow.com/questions/14051064/add-scalar-to-vector-in-blas-cublas-cuda
    const IdxType nblks = ceildiv(len, (IdxType)TPB);
    subtract_dev_scalar_kernel<math_t> <<<nblks, TPB, 0, stream>>>(outDev, inDev, singleScalarDev, len);
    CUDA_CHECK(cudaPeekAtLastError());
}

}; // end namespace LinAlg
}; // end namespace MLCommon
