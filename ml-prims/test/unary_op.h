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
#include "linalg/unary_op.h"


namespace MLCommon {
namespace LinAlg {


template <typename Type, typename IdxType>
__global__ void naiveScaleKernel(Type *out, const Type *in, Type scalar,
                                 IdxType len) {
  IdxType idx = threadIdx.x + ((IdxType)blockIdx.x * (IdxType)blockDim.x);
  if (idx < len) {
    out[idx] = scalar * in[idx];
  }
}

template <typename Type, typename IdxType = int>
void naiveScale(Type *out, const Type *in, Type scalar, int len) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  naiveScaleKernel<Type><<<nblks, TPB>>>(out, in, scalar, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T, typename IdxType = int>
struct UnaryOpInputs {
  T tolerance;
  IdxType len;
  T scalar;
  unsigned long long int seed;
};

template <typename T, typename IdxType = int>
::std::ostream &operator<<(::std::ostream &os,
                           const UnaryOpInputs<T,IdxType> &dims) {
  return os;
}

} // end namespace LinAlg
} // end namespace MLCommon
