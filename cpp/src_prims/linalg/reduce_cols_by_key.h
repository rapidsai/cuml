/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <stdlib.h>
#include <cub/cub.cuh>
#include <limits>
#include "cuda_utils.h"

namespace MLCommon {
namespace LinAlg {

///@todo: support col-major
///@todo: specialize this to support shared-mem based atomics

template <typename T, typename KeyType, typename IdxType>
__global__ void reduce_cols_by_key_kernel(const T* data, const KeyType* keys,
                                          T* out, IdxType nrows, IdxType ncols,
                                          IdxType nkeys) {
  IdxType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= (nrows * ncols))
    return;
  ///@todo: yikes! use fast-int-div!
  IdxType colId = idx % ncols;
  IdxType rowId = idx / ncols;
  KeyType key = keys[colId];
  atomicAdd(out+rowId*nkeys+key, data[idx]);
}

/**
 * @brief Computes the sum-reduction of matrix columns for each given key
 * @tparam T the input data type (as well as the output reduced matrix)
 * @tparam KeyType data type of the keys
 * @tparam IdxType indexing arithmetic type
 * @param data the input data (dim = nrows x ncols). This is assumed to be in
 * row-major layout
 * @param keys keys array (len = ncols). It is assumed that each key in this
 * array is between [0, nkeys). In case this is not true, the caller is expected
 * to have called make_monotonic primitive to prepare such a contiguous and
 * monotonically increasing keys array.
 * @param out the output reduced matrix along columns (dim = nrows x nkeys).
 * This will be assumed to be in row-major layout
 * @param nrows number of rows in the input data
 * @param ncols number of colums in the input data
 * @param nkeys number of unique keys in the keys array
 * @param stream cuda stream to launch the kernel onto
 */
template <typename T, typename KeyType = int, typename IdxType = int>
void reduce_cols_by_key(const T* data, const KeyType* keys, T* out,
                        IdxType nrows, IdxType ncols, IdxType nkeys,
                        cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(T) * nrows * nkeys, stream));
  constexpr int TPB = 256;
  int nblks = (int)ceildiv<IdxType>(nrows * ncols, TPB);
  reduce_cols_by_key_kernel<T, KeyType, IdxType><<<nblks, TPB, 0, stream>>>(
    data, keys, out, nrows, ncols, nkeys);
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // end namespace LinAlg
};  // end namespace MLCommon
