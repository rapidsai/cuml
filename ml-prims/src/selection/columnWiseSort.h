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

#define ALIGN_BYTE 256
#define ALIGN_MEMORY(x) (x + ALIGN_BYTE - 1) & ~(ALIGN_BYTE - 1)

namespace MLCommon {
namespace Selection {

using namespace std;

template <typename InType>
__global__ void devLayoutIdx(InType *in, int n_cols, int totalElements) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int n = n_cols;

  if (idx < totalElements) {
    in[idx] = idx % n;
  }
}

template <typename T>
__global__ void devOffsetKernel(T *in, T value, int n_times) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n_times)
    in[idx] = idx * value;
}

template <typename    InType,
    typename    OutType,
    int         BLOCK_SIZE,
    int         ITEMS_PER_THREAD>
__global__ void devKeyValSortColumnPerRow(const InType *inputKeys, InType *outputKeys, 
                                            OutType *inputVals, int n_rows, 
                                            int n_cols, InType MAX_VALUE) {
  
  typedef cub::BlockLoad<InType, BLOCK_SIZE, ITEMS_PER_THREAD, 
                          cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadTypeKey;
  typedef cub::BlockLoad<OutType, BLOCK_SIZE, ITEMS_PER_THREAD, 
                          cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadTypeVal;

  typedef cub::BlockRadixSort<InType, BLOCK_SIZE, ITEMS_PER_THREAD, 
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
  cub::StoreDirectStriped<BLOCK_SIZE>(threadIdx.x, inputVals + blockOffset, threadValues, n_cols);
  
  // @ToDo check correctness
  if (outputKeys)
    cub::StoreDirectStriped<BLOCK_SIZE>(threadIdx.x, outputKeys + blockOffset, threadKeys, n_cols);
}

template <typename OutType>
cudaError_t layoutIdx(OutType *in, int n_rows, int n_columns, cudaStream_t stream) {
  int totalElements = n_rows * n_columns;
  dim3 block(64);
  dim3 grid((totalElements + block.x - 1) / block.x );
  devLayoutIdx<OutType><<<grid, block, 0, stream>>>(in, n_columns, totalElements);
  return cudaGetLastError();
}

template <typename T>
cudaError_t layoutSortOffset(T *in, T value, int n_times, cudaStream_t stream) {
  dim3 block(64);
  dim3 grid((n_times + block.x - 1) / block.x);
  devOffsetKernel<T><<<grid, block, 0, stream>>>(in, value, n_times);
  return cudaGetLastError();
}

template <typename InType, typename OutType>
void sortColumnsPerRow(const InType *in, OutType *out, int n_rows, int n_columns, bool colMajor, 
                        bool &allocWorkspace, void *workspacePtr, size_t &workspaceSize, InType *sortedKeys=NULL,
                        bool returnKeys=false, cudaStream_t stream=0) {
  // assume non-square row-major matrices
  // 
  // current use-case: KNN, trustworthiness scores
  // output : either sorted indices or sorted indices and input values
  // future : this prim can be modified to be more generic and serve as a way to sort column entries per row
  //          i.e. another output format: sorted values only         

  int totalElements = n_rows * n_columns;
  // for current use-case sort is used in key-value sort mode. Key being the input distance

  int currDevice = 0, smemLimit = 0;
  CUDA_CHECK(cudaGetDevice(&currDevice));
  CUDA_CHECK(cudaDeviceGetAttribute(&smemLimit, cudaDevAttrMaxSharedMemoryPerBlock, currDevice));
  
  // assuming key-value sort for now - smem computation will change for value only sort

  // for 48KB smem/block, can fit in 6144 4byte key-value pair
  size_t perElementSmemUsage = sizeof(InType) + sizeof(OutType);
  size_t maxElementsForBlockSort = smemLimit / perElementSmemUsage;

  if (n_columns <= maxElementsForBlockSort) {
    
    // layout values
    CUDA_CHECK(layoutIdx(out, n_rows, n_columns, stream));
    // more elements per thread --> more register pressure
    // 512(blockSize) * 8 elements per thread = 71 register / thread
    dim3 gridK(n_rows);

    // instantiate some kernel combinations
    if (n_columns <= 512)
      devKeyValSortColumnPerRow<InType, OutType, 128, 4><<<gridK, 128, 0, stream>>>
                                (in, sortedKeys, out, n_rows, n_columns, numeric_limits<InType>::max());
    else if(n_columns > 512 && n_columns <= 1024 )
      devKeyValSortColumnPerRow<InType, OutType, 128, 8><<<gridK, 128, 0, stream>>>
                                (in, sortedKeys, out, n_rows, n_columns, numeric_limits<InType>::max());
    else if(n_columns > 1024 && n_columns <= 2048 )
      devKeyValSortColumnPerRow<InType, OutType, 256, 8><<<gridK, 256, 0, stream>>>
                                (in, sortedKeys, out, n_rows, n_columns, numeric_limits<InType>::max());
    else if(n_columns > 2048 && n_columns <= 4096 )
      devKeyValSortColumnPerRow<InType, OutType, 512, 8><<<gridK, 512, 0, stream>>>
                                (in, sortedKeys, out, n_rows, n_columns, numeric_limits<InType>::max());
    else
      devKeyValSortColumnPerRow<InType, OutType, 512, 12><<<gridK, 512, 0, stream>>>
                                  (in, sortedKeys, out, n_rows, n_columns, numeric_limits<InType>::max());
  }
  else if (n_columns > maxElementsForBlockSort && n_columns <= (1 << 16)) {
      // device Segmented radix sort
      // 65k column entries are arbitrary at the moment- need experiments
      int numSegments = n_rows + 1;

      // need auxillary storage: cub sorting + keys (if user not passing) + 
      // staging for values out + segment partition
      if (workspaceSize == 0) {
        OutType *tmpValIn = NULL;
        int *tmpOffsetBuffer = NULL;

        // first call is to get size of workspace
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(workspacePtr, workspaceSize, in, sortedKeys, 
                                                  tmpValIn, out, totalElements, numSegments, tmpOffsetBuffer, 
                                                  tmpOffsetBuffer+1));
        allocWorkspace = true;
        // more staging space for temp output of keys
        if (!sortedKeys)
          workspaceSize += ALIGN_MEMORY(sizeof(InType) * (size_t)totalElements);
        
        // value in KV pair need to be passed in, out buffer is separate
        workspaceSize += ALIGN_MEMORY(sizeof(OutType) * (size_t)totalElements);

        // for segment offsets
        workspaceSize += ALIGN_MEMORY(sizeof(int) * (size_t)numSegments);
        // can get away with unsigned short here if we stay at max 65k columns for this sort conditional
      }
      else {
          
          size_t workspaceOffset = 0;

          if (!sortedKeys) {
            sortedKeys = reinterpret_cast<InType *>(workspacePtr);
            workspaceOffset = ALIGN_MEMORY(sizeof(InType) * (size_t)totalElements);
            workspacePtr = (void *)((size_t)workspacePtr + workspaceOffset);
            workspaceSize  -= workspaceOffset;
          }

          
          OutType *dValuesIn = reinterpret_cast<OutType *>(workspacePtr);
          workspaceOffset = ALIGN_MEMORY(sizeof(OutType) * (size_t)totalElements);
          workspaceSize  -= workspaceOffset;
          workspacePtr = (void *)((size_t)workspacePtr + workspaceOffset);

          int *dSegmentOffsets = reinterpret_cast<int *>(workspacePtr);
          workspaceOffset = ALIGN_MEMORY(sizeof(int) * (size_t)numSegments);
          workspaceSize  -= workspaceOffset;
          workspacePtr = (void *)((size_t)workspacePtr + workspaceOffset);

          // layout idx
          CUDA_CHECK(layoutIdx(dValuesIn, n_rows, n_columns, stream));

          // layout segment lengths - spread out column length
          CUDA_CHECK(layoutSortOffset(dSegmentOffsets, n_columns, numSegments, stream));

          CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(workspacePtr, workspaceSize, in, sortedKeys, 
                                                  dValuesIn, out, totalElements, numSegments, dSegmentOffsets,
                                                  dSegmentOffsets+1, 0, sizeof(InType)*8, stream));
      }

  }
  else {
      // batched per row device wide sort
      // cub::DeviceRadixSort::SortPairs
      if (workspaceSize == 0) {
        OutType *tmpValIn = NULL;

        // first call is to get size of workspace
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(workspacePtr, workspaceSize, in, sortedKeys, 
                                                  tmpValIn, out, n_columns));
        allocWorkspace = true;

        if (!sortedKeys)
          workspaceSize += ALIGN_MEMORY(sizeof(InType) * (size_t)n_columns);

        workspaceSize += ALIGN_MEMORY(sizeof(OutType) * (size_t)n_columns);
      }
      else {

        size_t workspaceOffset = 0;

        if (!sortedKeys) {
          sortedKeys = reinterpret_cast<InType *>(workspacePtr);
          workspaceOffset = ALIGN_MEMORY(sizeof(InType) * (size_t)n_columns);
          workspacePtr = (void *)((size_t)workspacePtr + workspaceOffset);
          workspaceSize  -= workspaceOffset;
        }

        OutType *dValuesIn = reinterpret_cast<OutType *>(workspacePtr);
        workspaceOffset = ALIGN_MEMORY(sizeof(OutType) * (size_t)n_columns);
        workspaceSize  -= workspaceOffset;
        workspacePtr = (void *)((size_t)workspacePtr + workspaceOffset);

        // layout idx
        CUDA_CHECK(layoutIdx(dValuesIn, 1, n_columns, stream));

        for (int i = 0; i < n_rows; i++) {
          InType *rowIn = reinterpret_cast<InType*>((size_t)in + (i * sizeof(InType) * (size_t)n_columns));
          OutType *rowOut = reinterpret_cast<OutType*>((size_t)out + (i * sizeof(OutType) * (size_t)n_columns));

          CUDA_CHECK(cub::DeviceRadixSort::SortPairs(workspacePtr, workspaceSize, rowIn, sortedKeys,
                                                  dValuesIn, rowOut, n_columns));
        }
      }
  }
}
}; // end namespace Selection
}; // end namespace MLCommon
