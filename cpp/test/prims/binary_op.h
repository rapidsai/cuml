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
#include "linalg/binary_op.h"

namespace MLCommon {
namespace LinAlg {

template <typename InType, typename OutType, typename IdxType>
__global__ void naiveAddKernel(OutType *out, const InType *in1,
                               const InType *in2, IdxType len) {
  IdxType idx = threadIdx.x + ((IdxType)blockIdx.x * (IdxType)blockDim.x);
  if (idx < len) {
    out[idx] = static_cast<OutType>(in1[idx] + in2[idx]);
  }
}

template <typename InType, typename IdxType = int, typename OutType = InType>
void naiveAdd(OutType *out, const InType *in1, const InType *in2, IdxType len) {
  static const IdxType TPB = 64;
  IdxType nblks = ceildiv(len, TPB);
  naiveAddKernel<InType, OutType, IdxType><<<nblks, TPB>>>(out, in1, in2, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename InType, typename IdxType = int, typename OutType = InType>
struct BinaryOpInputs {
  InType tolerance;
  IdxType len;
  unsigned long long int seed;
};

template <typename InType, typename IdxType = int, typename OutType = InType>
::std::ostream &operator<<(::std::ostream &os,
                           const BinaryOpInputs<InType, IdxType, OutType> &d) {
  return os;
}

}  // end namespace LinAlg
}  // end namespace MLCommon
