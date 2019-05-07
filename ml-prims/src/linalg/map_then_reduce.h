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
#include "vectorized.h"


namespace MLCommon {
namespace LinAlg {

template<typename Type, int TPB> __device__ void reduce(Type *out, const Type acc){
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type tmp = BlockReduce(temp_storage).Sum(acc);
  if (threadIdx.x == 0) {
    myAtomicAdd(out, tmp);
  }
}

template <typename Type, typename MapOp, int TPB, typename ... Args>
__global__ void mapThenSumReduceKernel(Type *out, size_t len,
                                       MapOp map, const Type *in, Args... args) {
  Type acc = (Type)0;
  auto idx = (threadIdx.x + (blockIdx.x * blockDim.x));
  
  if (idx < len) {
      acc = map(in[idx], args[idx]...);
  }

  __syncthreads();

  reduce<Type, TPB>(out, acc);
}

template <typename Type, typename MapOp, int TPB, typename ... Args>
void mapThenSumReduceImpl(Type *out, size_t len, MapOp map,
                          cudaStream_t stream, const Type *in, Args ... args) {
  CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(Type), stream));
  const int nblks = ceildiv(len, (size_t)TPB);
  mapThenSumReduceKernel<Type, MapOp, TPB, Args...><<<nblks, TPB, 0, stream>>>(
    out, len, map, in, args...);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief CUDA version of map and then sum reduction operation
 * @tparam Type data-type upon which the math operation will be performed
 * @tparam MapOp the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam Args additional parameters
 * @param out the output sum-reduced value (assumed to be a device pointer)
 * @param len number of elements in the input array
 * @param map the device-lambda
 * @param stream cuda-stream where to launch this kernel
 * @param in the input array
 * @param args additional input arrays
 */

template <typename Type, typename MapOp, int TPB=256, typename ... Args>
void mapThenSumReduce(Type *out, size_t len, MapOp map,
                      cudaStream_t stream, const Type *in, Args... args) {
    mapThenSumReduceImpl<Type, MapOp, TPB, Args...>(out, len, map, stream, in, args...);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
