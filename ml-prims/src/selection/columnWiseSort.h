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

#include <cuda_runtime.h>
#include <cstddef>
#include "cuda_utils.h"

#include <cub/cub.cuh>
#include <limits>

namespace MLCommon {
namespace Selection {

using namespace std;

template <typename InType>
__global__ void layoutIdx(InType *in, int n_rows, int n_cols) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int m = n_rows, n = n_cols;

  if (idx < m * n) {
    in[idx] = idx % n;
  }
}

template <typename    InType,
    typename    OutType,
    int         blockThreads,
    int         ITEMS_PER_THREAD>
__global__ void devKeyValSortColumnPerRow(const InType *inputKeys, OutType *inputVals, int n_rows, 
                                      int n_cols, InType MAX_VALUE) {
  
  typedef cub::BlockLoad<InType, blockThreads, ITEMS_PER_THREAD, 
                          cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadTypeKey;
  typedef cub::BlockLoad<OutType, blockThreads, ITEMS_PER_THREAD, 
                          cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadTypeVal;

  typedef cub::BlockRadixSort<InType, blockThreads, ITEMS_PER_THREAD, 
                                OutType> BlockRadixSortType;

  __shared__ union TempStorage
  {
    struct {
      typename BlockLoadTypeKey::TempStorage keyLoad;
      typename BlockLoadTypeVal::TempStorage valLoad;
    } load;
    typename BlockRadixSortType::TempStorage sort;
  } tempStorage;

  InType threadKeys[ITEMS_PER_THREAD];
  OutType threadValues[ITEMS_PER_THREAD];

  int blockOffset = blockIdx.x * n_cols;
  BlockLoadTypeKey(tempStorage.load.keyLoad).Load(inputKeys + blockOffset, threadKeys, n_cols, MAX_VALUE);
  BlockLoadTypeVal(tempStorage.load.valLoad).Load(inputVals + blockOffset, threadValues, n_cols, MAX_VALUE);

  __syncthreads();

  BlockRadixSortType(tempStorage.sort).SortBlockedToStriped(threadKeys, threadValues);

  // storing index values back (not keys)
  cub::StoreDirectStriped<blockThreads>(threadIdx.x, inputVals + blockOffset, threadValues, n_cols);
  
}

template <typename InType, typename OutType>
void sortColumnsPerRow(const InType *in, OutType *out, int n_rows, int n_columns, bool colMajor, 
                  bool returnDistance, void *workspace, size_t workspaceSize, 
                  cudaStream_t stream=0) {
  // assume non-square row-major matrices
  // 
  // current use-case: KNN, trustworthiness scores
  // output : either sorted indices or sorted indices and input values
  // future : this prim can be modified to be more generic and serve as a way to sort column entries per row
  //          i.e. another output format: sorted values only         

  // check alloc space of workspace ?

  auto totalElements = n_rows * n_columns;
  // for current use-case sort is used in key-value sort mode. Key being the input distance
  // layout values
  dim3 block(128);
  dim3 grid((totalElements + block.x - 1) / block.x );
  layoutIdx<OutType><<<grid, block, 0, stream>>>(out, n_rows, n_columns);
  
  int currDevice = 0, smemLimit = 0;
  CUDA_CHECK(cudaGetDevice(&currDevice));
  CUDA_CHECK(cudaDeviceGetAttribute(&smemLimit, cudaDevAttrMaxSharedMemoryPerBlock, currDevice));
  
  // assuming key-value sort for now - smem computation will change for value only sort

  // for 48KB smem/block, can fit in 6144 4byte key-value pair
  size_t perElementSmemUsage = sizeof(InType) + sizeof(OutType);
  size_t maxElementsForBlockSort = smemLimit / perElementSmemUsage;

  if (n_columns <= maxElementsForBlockSort) {
    // more elements per thread --> more register pressure
    // 512(blockSize) * 8 elements per thread = 71 register / thread
    dim3 gridK(n_rows);

    // instantiate some kernel combinations
    if (n_columns <= 512)
      devKeyValSortColumnPerRow<InType, OutType, 128, 4><<<gridK, 128, 0, stream>>>
                                (in, out, n_rows, n_columns, numeric_limits<InType>::max());
    else if(n_columns > 512 && n_columns <= 1024 )
      devKeyValSortColumnPerRow<InType, OutType, 128, 8><<<gridK, 128, 0, stream>>>
                                (in, out, n_rows, n_columns, numeric_limits<InType>::max());
    else if(n_columns > 1024 && n_columns <= 2048 )
      devKeyValSortColumnPerRow<InType, OutType, 256, 8><<<gridK, 256, 0, stream>>>
                                (in, out, n_rows, n_columns, numeric_limits<InType>::max());
    else if(n_columns > 2048 && n_columns <= 4096 )
      devKeyValSortColumnPerRow<InType, OutType, 512, 8><<<gridK, 512, 0, stream>>>
                                (in, out, n_rows, n_columns, numeric_limits<InType>::max());
    else
      devKeyValSortColumnPerRow<InType, OutType, 512, 12><<<gridK, 512, 0, stream>>>
                                  (in, out, n_rows, n_columns, numeric_limits<InType>::max());
  }
  else if (n_columns > maxElementsForBlockSort && n_columns <= (1 << 16)) {
      // device Segmented radix sort
      // cub::DeviceSegmentedRadixSort::SortPairs
      // 65k column entries are arbitrary at the moment- need experiments

  }
  else {
      // batched per row device wide sort
      // cub::DeviceRadixSort::SortPairs
  }
}
}; // end namespace Selection
}; // end namespace MLCommon
