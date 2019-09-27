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

template <typename InType, int VecLen, typename Lambda, typename IdxType,
          typename OutType>
__global__ void binaryOpKernel(OutType *out, const InType *in1,
                               const InType *in2, IdxType len, Lambda op) {
  typedef TxN_t<InType, VecLen> InVecType;
  typedef TxN_t<OutType, VecLen> OutVecType;
  InVecType a, b;
  OutVecType c;
  IdxType idx = threadIdx.x + ((IdxType)blockIdx.x * blockDim.x);
  idx *= InVecType::Ratio;
  if (idx >= len) return;
  a.load(in1, idx);
  b.load(in2, idx);
#pragma unroll
  for (int i = 0; i < InVecType::Ratio; ++i) {
    c.val.data[i] = op(a.val.data[i], b.val.data[i]);
  }
  c.store(out, idx);
}

template <typename InType, int VecLen, typename Lambda, typename IdxType,
          typename OutType, int TPB>
void binaryOpImpl(OutType *out, const InType *in1, const InType *in2,
                  IdxType len, Lambda op, cudaStream_t stream) {
  const IdxType nblks = ceildiv(VecLen ? len / VecLen : len, (IdxType)TPB);
  binaryOpKernel<InType, VecLen, Lambda, IdxType, OutType>
    <<<nblks, TPB, 0, stream>>>(out, in1, in2, len, op);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief perform element-wise binary operation on the input arrays
 * @tparam InType input data-type
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam OutType output data-type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in1 the first input array
 * @param in2 the second input array
 * @param len number of elements in the input array
 * @param op the device-lambda
 * @param stream cuda stream where to launch work
 * @note Lambda must be a functor with the following signature:
 *       `OutType func(const InType& val1, const InType& val2);`
 */
template <typename InType, typename Lambda, typename OutType = InType,
          typename IdxType = int, int TPB = 256>
void binaryOp(OutType *out, const InType *in1, const InType *in2, IdxType len,
              Lambda op, cudaStream_t stream) {
  constexpr auto maxSize =
    sizeof(InType) > sizeof(OutType) ? sizeof(InType) : sizeof(OutType);
  size_t bytes = len * maxSize;
  if (16 / maxSize && bytes % 16 == 0) {
    binaryOpImpl<InType, 16 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (8 / maxSize && bytes % 8 == 0) {
    binaryOpImpl<InType, 8 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (4 / maxSize && bytes % 4 == 0) {
    binaryOpImpl<InType, 4 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (2 / maxSize && bytes % 2 == 0) {
    binaryOpImpl<InType, 2 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (1 / maxSize) {
    binaryOpImpl<InType, 1 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else {
    binaryOpImpl<InType, 1, Lambda, IdxType, OutType, TPB>(out, in1, in2, len,
                                                           op, stream);
  }
}

};  // end namespace LinAlg
};  // end namespace MLCommon
