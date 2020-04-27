/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
#include "linalg/unary_op.h"

namespace MLCommon {
namespace LinAlg {

template <typename InType, typename OutType, typename IdxType>
__global__ void naiveScaleKernel(OutType *out, const InType *in, InType scalar,
                                 IdxType len) {
  IdxType idx = threadIdx.x + ((IdxType)blockIdx.x * (IdxType)blockDim.x);
  if (idx < len) {
    if (in == nullptr) {
      // used for testing writeOnlyUnaryOp
      out[idx] = static_cast<OutType>(scalar * idx);
    } else {
      out[idx] = static_cast<OutType>(scalar * in[idx]);
    }
  }
}

template <typename InType, typename IdxType = int, typename OutType = InType>
void naiveScale(OutType *out, const InType *in, InType scalar, int len,
                cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  naiveScaleKernel<InType, OutType, IdxType>
    <<<nblks, TPB, 0, stream>>>(out, in, scalar, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename InType, typename IdxType = int, typename OutType = InType>
struct UnaryOpInputs {
  OutType tolerance;
  IdxType len;
  InType scalar;
  unsigned long long int seed;
};

template <typename InType, typename IdxType = int, typename OutType = InType>
::std::ostream &operator<<(::std::ostream &os,
                           const UnaryOpInputs<InType, IdxType, OutType> &d) {
  return os;
}

}  // end namespace LinAlg
}  // end namespace MLCommon
