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

// Kernel (based on norm.h) to perform reductions along the coalesced dimension
// of the matrix, i.e. reduce along rows for row major or reduce along columns
// for column major layout. Kernel does an inplace reduction adding to original
// values of dots.
template <typename Type, int TPB, typename MainLambda, typename Lambda>
__global__ void coalescedReductionKernel(Type *dots, const Type *data, int D,
                                         int N, MainLambda main_op,
                                         Lambda fin_op, bool inplace) {
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type thread_data = Type(0);
  int rowStart = blockIdx.x * D;
  for (int i = threadIdx.x; i < D; i += TPB) {
    int idx = rowStart + i;
    thread_data += main_op(data[idx]);
  }
  Type acc = BlockReduce(temp_storage).Sum(thread_data);
  if (threadIdx.x == 0) {
    if(inplace) {
      dots[blockIdx.x] += fin_op(acc);
    } else {
      dots[blockIdx.x] = fin_op(acc);
    }
  }
}


/**
 * @brief Compute reduction of the input matrix along the leading dimension
 *
 * @tparam Type the data type
 * @tparam MainLambda Lambda applied while acculumation (eg: L1 or L2 norm)
 * @tparam Lambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * @param dots the output reduction vector
 * @param data the input matrix
 * @param D leading dimension of data
 * @param N second dimension data
 * @param main_op elementwise operation to apply before reduction
 * @param fin_op elementwise operation to apply before storing results
 * @param inplace reduction result added inplace or overwrites old values?
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename MainLambda, typename Lambda>
void coalesedReduction(Type *dots, const Type *data, int D, int N,
                       MainLambda main_op, Lambda fin_op, bool inplace = false,
                       cudaStream_t stream = 0) {
  // One block per reduction
  // Efficient only for large leading dimensions
  if (D <= 32) {
    coalescedReductionKernel<Type,  32><<<N,  32, 0, stream>>>
      (dots, data, D, N, main_op, fin_op, inplace);
  } else if (D <= 64) {
    coalescedReductionKernel<Type,  64><<<N,  64, 0, stream>>>
      (dots, data, D, N, main_op, fin_op, inplace);
  } else if (D <= 128) {
    coalescedReductionKernel<Type, 128><<<N, 128, 0, stream>>>
      (dots, data, D, N, main_op, fin_op, inplace);
  } else {
    coalescedReductionKernel<Type, 256><<<N, 256, 0, stream>>>
      (dots, data, D, N, main_op, fin_op, inplace);
  }
}

}; // end namespace LinAlg
}; // end namespace MLCommon
