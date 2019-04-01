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


template <typename Type>
__global__ void naiveScaleKernel(Type *out, const Type *in, Type scalar,
                                 int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = scalar * in[idx];
  }
}

template <typename Type>
void naiveScale(Type *out, const Type *in, Type scalar, int len, cudaStream_t stream=0) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  naiveScaleKernel<Type><<<nblks, TPB, 0, stream>>>(out, in, scalar, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
struct UnaryOpInputs {
  T tolerance;
  int len;
  T scalar;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const UnaryOpInputs<T> &dims) {
  return os;
}

} // end namespace LinAlg
} // end namespace MLCommon
