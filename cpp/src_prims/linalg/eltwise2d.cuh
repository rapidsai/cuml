/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cuml/common/utils.hpp>

namespace MLCommon {
namespace LinAlg {

template <typename Type, typename Lambda>
CUML_KERNEL void eltwise2DKernel(int rows,  // m
                                 int cols,  // n
                                 const Type* dotA,
                                 const Type* dotB,
                                 const Type* pC,
                                 Type* pD,
                                 Type alpha,
                                 Type beta,
                                 Lambda op)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cols * rows) {
    const auto x  = tid % cols;
    const auto y  = tid / cols;
    const auto ab = pD[tid];
    const auto a  = dotA[y];
    const auto b  = dotB[x];
    Type accm     = alpha * op(a, b, ab);

    if (beta) { accm += beta * pC[tid]; }
    pD[tid] = accm;
  }
}

template <typename Type, typename Lambda>
void eltwise2D(int rows,  // m
               int cols,  // n
               const Type* dotA,
               const Type* dotB,
               const Type* pC,
               Type* pD,
               Type alpha,
               Type beta,
               Lambda op,
               cudaStream_t stream)
{
  size_t threads = 256;
  size_t blocks  = ((cols * rows) + threads - 1) / threads;
  eltwise2DKernel<Type>
    <<<blocks, threads, 0, stream>>>(rows, cols, dotA, dotB, pC, pD, alpha, beta, op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace LinAlg
}  // namespace MLCommon
