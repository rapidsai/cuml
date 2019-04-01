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
#include "linalg/unary_op.h"
#include <type_traits>


namespace MLCommon {
namespace LinAlg {

// Kernel to perform reductions along the strided dimension
// of the matrix, i.e. reduce along columns for row major or reduce along rows
// for column major layout
template <typename Type, typename MainLambda>
__global__ void stridedSummationKernel(Type *dots, const Type *data, int D, int N, Type init,
                                       MainLambda main_op) {
  // Thread reduction
  Type thread_data = Type(init);
  int colStart = blockIdx.x * blockDim.x + threadIdx.x;
  if (colStart < D) {
    int rowStart = blockIdx.y * blockDim.y + threadIdx.y;
    int stride = blockDim.y * gridDim.y;
    for (int j = rowStart; j < N; j += stride) {
      int idx = colStart + j*D;
      thread_data += main_op(data[idx], j);
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

// Kernel to perform reductions along the strided dimension
// of the matrix, i.e. reduce along columns for row major or reduce along rows
// for column major layout
template <typename InType, typename OutType, typename IdxType,
          typename MainLambda, typename ReduceLambda>
__global__ void stridedReductionKernel(OutType *dots, const InType *data, int D, int N, OutType init,
                                       MainLambda main_op, ReduceLambda reduce_op) {
  // Thread reduction
  OutType thread_data = init;
  IdxType colStart = blockIdx.x * blockDim.x + threadIdx.x;
  if (colStart < D) {
    IdxType rowStart = blockIdx.y * blockDim.y + threadIdx.y;
    IdxType stride = blockDim.y * gridDim.y;
    for (IdxType j = rowStart; j < N; j += stride) {
      IdxType idx = colStart + j*D;
      thread_data = reduce_op(thread_data, main_op(data[idx], j));
    }
  }

  // Block reduction
  extern __shared__ char tmp[]; // One element per thread in block
  auto *temp = (OutType *)tmp; // Cast to desired type
  int myidx = threadIdx.x + blockDim.x * threadIdx.y;
  temp[myidx] = thread_data;
  __syncthreads();
  for (int j = blockDim.y / 2; j > 0; j /= 2) {
    if (threadIdx.y < j)
      temp[myidx] = reduce_op(temp[myidx], temp[myidx + j*blockDim.x]);
    __syncthreads();
  }

  // Grid reduction
  if ( (colStart < D) && (threadIdx.y == 0) )
    myAtomicReduce(dots+colStart, temp[myidx], reduce_op);
}


/**
 * @brief Compute reduction of the input matrix along the strided dimension
 *
 * @tparam InType the data type of the input
 * @tparam OutType the data type of the output (as well as the data type for
 *  which reduction is performed)
 * @tparam IdxType data type of the indices of the array
 * @tparam MainLambda Unary lambda applied while acculumation (eg: L1 or L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*MainLambda)(InType, IdxType);</pre>
 * @tparam ReduceLambda Binary lambda applied for reduction (eg: addition(+) for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*ReduceLambda)(OutType);</pre>
 * @tparam FinalLambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*FinalLambda)(OutType);</pre>
 * @param dots the output reduction vector
 * @param data the input matrix
 * @param D leading dimension of data
 * @param N second dimension data
 * @param init initial value to use for the reduction
 * @param main_op elementwise operation to apply before reduction
 * @param reduce_op binary reduction operation
 * @param final_op elementwise operation to apply before storing results
 * @param inplace reduction result added inplace or overwrites old values?
 * @param stream cuda stream where to launch work
 */
template <typename InType, typename OutType = InType, typename IdxType = int,
          typename MainLambda = Nop<InType, IdxType>,
          typename ReduceLambda = Sum<OutType>,
          typename FinalLambda = Nop<OutType>>
void stridedReduction(OutType *dots, const InType *data, int D, int N, OutType init,
                      cudaStream_t stream, bool inplace = false,
                      MainLambda main_op = Nop<InType, IdxType>(),
                      ReduceLambda reduce_op = Sum<OutType>(),
                      FinalLambda final_op = Nop<OutType>()) {
  ///@todo: this extra should go away once we have eliminated the need
  /// for atomics in stridedKernel (redesign for this is already underway)
  if (!inplace)
    unaryOp(dots, dots, D,
            [init] __device__ (OutType a) { return init; }, stream);

  // Arbitrary numbers for now, probably need to tune
  const dim3 thrds(32, 16);
  int elemsPerThread = ceildiv(N, (int)thrds.y);
  elemsPerThread = (elemsPerThread > 8) ? 8 : elemsPerThread;
  const dim3 nblks(ceildiv(D, (int)thrds.x), ceildiv(N, (int)thrds.y * elemsPerThread));
  const int shmemSize = sizeof(OutType) * thrds.x * thrds.y;

  ///@todo: this complication should go away once we have eliminated the need
  /// for atomics in stridedKernel (redesign for this is already underway)
  if (std::is_same<ReduceLambda, Sum<OutType>>::value &&
      std::is_same<InType, OutType>::value)
    stridedSummationKernel<InType><<<nblks, thrds, shmemSize, stream>>>(dots, data, D, N, init, main_op);
  else
    stridedReductionKernel<InType, OutType, IdxType>
      <<<nblks, thrds, shmemSize, stream>>>(dots, data, D, N, init, main_op, reduce_op);

  ///@todo: this complication should go away once we have eliminated the need
  /// for atomics in stridedKernel (redesign for this is already underway)
  // Perform final op on output data
  if (!std::is_same<FinalLambda, Nop<OutType>>::value)
    unaryOp(dots, dots, D, final_op, stream);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
