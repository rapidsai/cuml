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
#include "linalg/add.h"

namespace MLCommon {
namespace LinAlg {

template <typename InT, typename OutT = InT>
__global__ void naiveAddElemKernel(OutT *out, const InT *in1, const InT *in2,
                                   int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = OutT(in1[idx] + in2[idx]);
  }
}

template <typename InT, typename OutT = InT>
void naiveAddElem(OutT *out, const InT *in1, const InT *in2, int len) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  naiveAddElemKernel<InT, OutT><<<nblks, TPB>>>(out, in1, in2, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename InT, typename OutT = InT>
struct AddInputs {
  OutT tolerance;
  int len;
  unsigned long long int seed;
};

template <typename InT, typename OutT = InT>
::std::ostream &operator<<(::std::ostream &os,
                           const AddInputs<InT, OutT> &dims) {
  return os;
}

};  // end namespace LinAlg
};  // end namespace MLCommon
