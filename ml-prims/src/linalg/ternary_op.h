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

template <typename math_t, int veclen_, typename Lambda>
__global__ void ternaryOpKernel(math_t *out, 
                                const math_t *in1, const math_t *in2, const math_t *in3,
                                int len, Lambda op) {
  typedef TxN_t<math_t, veclen_> VecType;
  VecType a, b, c;
  int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * VecType::Ratio;
  if (idx >= len)
    return;
  a.load(in1, idx);
  b.load(in2, idx);
  c.load(in3, idx);
#pragma unroll
  for (int i = 0; i < VecType::Ratio; ++i) {
    a.val.data[i] = op(a.val.data[i], b.val.data[i], c.val.data[i]);
  }
  a.store(out, idx);
}

template <typename math_t, int veclen_, typename Lambda, int TPB>
void ternaryOpImpl(math_t *out, 
                    const math_t *in1, const math_t *in2, const math_t *in3,
                    int len, Lambda op, cudaStream_t stream) {
  const int nblks = ceildiv(veclen_ ? len / veclen_ : len, TPB);
  ternaryOpKernel<math_t, veclen_, Lambda><<<nblks, TPB, 0, stream>>>(
    out, in1, in2, in3, len, op);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief perform element-wise ternary operation on the input arrays
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in1 the first input array
 * @param in2 the second input array
 * @param in3 the third input array
 * @param len number of elements in the input array
 * @param op the device-lambda
 * @param stream cuda stream where to launch work
 */
template <typename math_t, typename Lambda, int TPB = 256>
void ternaryOp(math_t *out, 
                const math_t *in1, const math_t *in2, const math_t *in3,
                int len, Lambda op, cudaStream_t stream) {
  size_t bytes = len * sizeof(math_t);
  if (16 / sizeof(math_t) && bytes % 16 == 0) {
    ternaryOpImpl<math_t, 16 / sizeof(math_t), Lambda, TPB>(out, in1, in2, in3, len,
                                                           op, stream);
  } else if (8 / sizeof(math_t) && bytes % 8 == 0) {
    ternaryOpImpl<math_t, 8 / sizeof(math_t), Lambda, TPB>(out, in1, in2, in3, len,
                                                          op, stream);
  } else if (4 / sizeof(math_t) && bytes % 4 == 0) {
    ternaryOpImpl<math_t, 4 / sizeof(math_t), Lambda, TPB>(out, in1, in2, in3, len,
                                                          op, stream);
  } else if (2 / sizeof(math_t) && bytes % 2 == 0) {
    ternaryOpImpl<math_t, 2 / sizeof(math_t), Lambda, TPB>(out, in1, in2, in3, len,
                                                          op, stream);
  } else if (1 / sizeof(math_t)) {
    ternaryOpImpl<math_t, 1 / sizeof(math_t), Lambda, TPB>(out, in1, in2, in3, len,
                                                          op, stream);
  } else {
    ternaryOpImpl<math_t, 1, Lambda, TPB>(out, in1, in2, in3, len, op, stream);
  }
}


}; // end namespace LinAlg
}; // end namespace MLCommon
