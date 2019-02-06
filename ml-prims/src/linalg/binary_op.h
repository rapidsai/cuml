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
__global__ void binaryOpKernel(math_t *out, const math_t *in1,
                               const math_t *in2, int len, Lambda op) {
  typedef TxN_t<math_t, veclen_> VecType;
  VecType a, b;
  int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * VecType::Ratio;
  if (idx >= len)
    return;
  a.load(in1, idx);
  b.load(in2, idx);
#pragma unroll
  for (int i = 0; i < VecType::Ratio; ++i) {
    a.val.data[i] = op(a.val.data[i], b.val.data[i]);
  }
  a.store(out, idx);
}

template <typename math_t, int veclen_, typename Lambda, int TPB>
void binaryOpImpl(math_t *out, const math_t *in1, const math_t *in2, int len,
                  Lambda op, cudaStream_t stream = 0) {
  const int nblks = ceildiv(veclen_ ? len / veclen_ : len, TPB);
  binaryOpKernel<math_t, veclen_, Lambda><<<nblks, TPB, 0, stream>>>(
    out, in1, in2, len, op);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief perform element-wise binary operation on the input arrays
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in1 the first input array
 * @param in2 the second input array
 * @param len number of elements in the input array
 * @param op the device-lambda
 * @param stream cuda stream where to launch work
 */
template <typename math_t, typename Lambda, int TPB = 256>
void binaryOp(math_t *out, const math_t *in1, const math_t *in2, int len,
              Lambda op, cudaStream_t stream = 0) {
  size_t bytes = len * sizeof(math_t);
  if (16 / sizeof(math_t) && bytes % 16 == 0) {
    binaryOpImpl<math_t, 16 / sizeof(math_t), Lambda, TPB>(out, in1, in2, len,
                                                           op, stream);
  } else if (8 / sizeof(math_t) && bytes % 8 == 0) {
    binaryOpImpl<math_t, 8 / sizeof(math_t), Lambda, TPB>(out, in1, in2, len,
                                                          op, stream);
  } else if (4 / sizeof(math_t) && bytes % 4 == 0) {
    binaryOpImpl<math_t, 4 / sizeof(math_t), Lambda, TPB>(out, in1, in2, len,
                                                          op, stream);
  } else if (2 / sizeof(math_t) && bytes % 2 == 0) {
    binaryOpImpl<math_t, 2 / sizeof(math_t), Lambda, TPB>(out, in1, in2, len,
                                                          op, stream);
  } else if (1 / sizeof(math_t)) {
    binaryOpImpl<math_t, 1 / sizeof(math_t), Lambda, TPB>(out, in1, in2, len,
                                                          op, stream);
  } else {
    binaryOpImpl<math_t, 1, Lambda, TPB>(out, in1, in2, len, op, stream);
  }
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
 * @param n_gpus number of gpus
 * @param op the device-lambda
 */
template <typename math_t, typename Lambda, int TPB = 256>
void binaryOpMG(TypeMG<math_t> *out, const TypeMG<math_t> *in1,
                const TypeMG<math_t> *in2, int len, int n_gpus, Lambda op,
                bool sync = false) {
  for (int i = 0; i < n_gpus; i++) {
    CUDA_CHECK(cudaSetDevice(in1[i].gpu_id));

    int len = in1[i].n_cols * in1[i].n_rows;
    binaryOp(out[i].d_data, in1[i].d_data, in2[i].d_data, len, op,
             in1[i].stream);
  }

  if (sync)
    streamSyncMG(in1, n_gpus);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
