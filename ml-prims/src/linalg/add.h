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


namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @param out the output buffer
 * @param in the input buffer
 * @param scalar the scalar used in the operations
 * @param len number of elements in the input buffer
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename math_t>
void addScalar(math_t *out, const math_t *in, math_t scalar, int len,
               cudaStream_t stream) {
  unaryOp(out, in, len,
          [scalar] __device__(math_t in) { return in + scalar; },
          stream);
}
/** @} */

/**
 * @defgroup BinaryOps Element-wise binary operations on the input buffers
 * @param out the output buffer
 * @param in1 the first input buffer
 * @param in2 the second input buffer
 * @param len number of elements in the input buffers
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename math_t>
void add(math_t *out, const math_t *in1, const math_t *in2, int len,
         cudaStream_t stream = 0) {
  binaryOp(out, in1, in2, len,
           [] __device__(math_t a, math_t b) { return a + b; }, stream);
}
/** @} */



template<class math_t>
__global__ void add_dev_scalar_kernel(math_t* outDev, const math_t* inDev, const math_t *singleScalarDev, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        outDev[i] = inDev[i] + *singleScalarDev;
    }
}

/** Substract single value pointed by singleScalarDev parameter in device memory from inDev[i] and write result to outDev[i]
 * @param out the output buffer
 * @param in the input buffer
 * @param singleScalarDev pointer to the scalar located in device memory
 * @param len number of elements in the input and output buffer
 * @{
 */
template <typename math_t>
void addDevScalar(math_t* outDev, const math_t* inDev, const math_t* singleScalarDev, int len)
{
    // TODO: block dimension has not been tuned
    dim3 block (256);
    dim3 grid((len + block.x - 1) / block.x);
    add_dev_scalar_kernel<math_t> <<<grid, block>>>(outDev, inDev, singleScalarDev, len);
    CUDA_CHECK(cudaPeekAtLastError());
}

}; // end namespace LinAlg
}; // end namespace MLCommon
