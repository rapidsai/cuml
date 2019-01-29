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

#include <cub/cub.cuh>
#include "cuda_utils.h"


namespace MLCommon {
namespace LinAlg {

// Kernel to perform reductions along the strided dimension
// of the matrix, i.e. reduce along columns for row major or reduce along rows
// for column major layout
template <typename Type, typename Lambda>
__global__ void stridedReductionKernel(Type *dots, const Type *data, int D, int N,
                                       Lambda op) {
  // Thread reduction
  Type thread_data = Type(0);
  int colStart = blockIdx.x * blockDim.x + threadIdx.x;
  if (colStart < D) {
    int rowStart = blockIdx.y * blockDim.y + threadIdx.y;
    int stride = blockDim.y * gridDim.y;
    for (int j = rowStart; j < N; j += stride) {
      int idx = colStart + j*D;
      thread_data += op(data[idx]);
    }
  }

  // Block reduction
  extern __shared__ char tmp[]; // One element per thread in block
  Type *temp = (Type *)tmp; // Cast to desired type
  int myidx = threadIdx.x + blockDim.x * threadIdx.y;
  temp[myidx] = thread_data;
  __syncthreads();
  for (int j = blockDim.y / 2; j > 0; j /= 2) {
    if (threadIdx.y < j)
      temp[myidx] += temp[myidx + j*blockDim.x];
    __syncthreads();
  }

  // Grid reduction
  if ( (colStart < D) && (threadIdx.y == 0) )
    myAtomicAdd(dots+colStart, temp[myidx]);
}


/**
 * @brief Compute reduction of the input matrix along the strided dimension
 *
 * @tparam Type the data type
 * @param dots the output reduction vector
 * @param data the input matrix
 * @param D leading dimension of data
 * @param N second dimension data
 * @param op elementwise operation to apply before reduction (Sum)
 * @param inplace reduction result added inplace or overwrites old values?
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda>
void stridedReduction(Type *dots, const Type *data, int D, int N, Lambda op,
          bool inplace = false, cudaStream_t stream = 0) {
  if (!inplace)
    CUDA_CHECK(cudaMemsetAsync(dots, 0, sizeof(Type) * D, stream));

  // Arbitrary numbers for now, probably need to tune
  const dim3 thrds(32, 16);
  int elemsPerThread = ceildiv(N, (int)thrds.y);
  elemsPerThread = (elemsPerThread > 8) ? 8 : elemsPerThread;
  const dim3 nblks(ceildiv(D, (int)thrds.x), ceildiv(N, (int)thrds.y * elemsPerThread));
  const int shmemSize = sizeof(Type) * thrds.x * thrds.y;
  stridedReductionKernel<Type><<<nblks,  thrds, shmemSize, stream>>>(dots, data, D, N, op);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
