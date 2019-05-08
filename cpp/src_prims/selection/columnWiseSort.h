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

#include <cuda_runtime.h>
#include <cstddef>
#include "cuda_utils.h"

#include <cub/cub.cuh>
#include <limits>

#define INST_BLOCK_SORT(keyIn, keyOut, valueInOut, rows, columns, blockSize, elemPT, stream) \
devKeyValSortColumnPerRow<InType, OutType, blockSize, elemPT><<<rows, blockSize, 0, stream>>> \
        (keyIn, keyOut, valueInOut, rows, columns, std::numeric_limits<InType>::max())

namespace MLCommon {
namespace Selection {

template <typename InType, int BLOCK_SIZE>
struct TemplateChecker {
  enum {
    IsValid = (std::is_same< InType, short >::value && BLOCK_SIZE <= 1024) || \
              (std::is_same< InType, int >::value && BLOCK_SIZE <= 1024) || \
              (std::is_same< InType, float >::value && BLOCK_SIZE <= 1024) || \
              (std::is_same< InType, double >::value && BLOCK_SIZE <= 512)
  };
};

template <typename InType, typename OutType, int BLOCK_SIZE, int ITEMS_PER_THREAD>
struct SmemPerBlock {
  typedef cub::BlockLoad<InType, BLOCK_SIZE, ITEMS_PER_THREAD,
                          cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadTypeKey;

  typedef cub::BlockRadixSort<InType, BLOCK_SIZE, ITEMS_PER_THREAD,
                                OutType> BlockRadixSortType;

  union TempStorage
  {
    typename BlockLoadTypeKey::TempStorage keyLoad;
    typename BlockRadixSortType::TempStorage sort;
  } tempStorage;
};

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

// block level radix sort - can only sort as much data we can fit within shared memory
template <
    typename    InType,
    typename    OutType,
    int         BLOCK_SIZE,
    int         ITEMS_PER_THREAD,
    typename    std::enable_if<TemplateChecker<InType, BLOCK_SIZE>::IsValid,
                               InType>::type* = nullptr>
__global__ void
__launch_bounds__(1024, 1) devKeyValSortColumnPerRow(const InType *inputKeys, InType *outputKeys,
                                            OutType *inputVals, int n_rows, 
                                            int n_cols, InType MAX_VALUE) {

  typedef cub::BlockLoad<InType, BLOCK_SIZE, ITEMS_PER_THREAD, 
                          cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadTypeKey;

  typedef cub::BlockRadixSort<InType, BLOCK_SIZE, ITEMS_PER_THREAD, 
                                OutType> BlockRadixSortType;

  __shared__ SmemPerBlock<InType, OutType, BLOCK_SIZE, ITEMS_PER_THREAD> tmpSmem;

  InType threadKeys[ITEMS_PER_THREAD];
  OutType threadValues[ITEMS_PER_THREAD];

  int blockOffset = blockIdx.x * n_cols;
  BlockLoadTypeKey(tmpSmem.tempStorage.keyLoad).Load(inputKeys + blockOffset, threadKeys, n_cols, MAX_VALUE);

  OutType idxBase = threadIdx.x * ITEMS_PER_THREAD;
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    OutType eId = idxBase + (OutType)i;
    if (eId < n_cols)
      threadValues[i] = eId;
    else
      threadValues[i] = MAX_VALUE;
  }

  __syncthreads();

  BlockRadixSortType(tmpSmem.tempStorage.sort).SortBlockedToStriped(threadKeys, threadValues);

  // storing index values back (not keys)
  cub::StoreDirectStriped<BLOCK_SIZE>(threadIdx.x, inputVals + blockOffset, threadValues, n_cols);

  if (outputKeys)
    cub::StoreDirectStriped<BLOCK_SIZE>(threadIdx.x, outputKeys + blockOffset, threadKeys, n_cols);
}

template <
    typename    InType,
    typename    OutType,
    int         BLOCK_SIZE,
    int         ITEMS_PER_THREAD,
    typename    std::enable_if<!(TemplateChecker<InType, BLOCK_SIZE>::IsValid), InType>::type* = nullptr>
__global__ void devKeyValSortColumnPerRow(const InType *inputKeys, InType *outputKeys, 
                                            OutType *inputVals, int n_rows, 
                                            int n_cols, InType MAX_VALUE) {
  // place holder function
  // so that compiler unrolls for all template types successfully
}


// helper function to layout values (index's) for key-value sort
template <typename OutType>
cudaError_t layoutIdx(OutType *in, int n_rows, int n_columns, cudaStream_t stream) {
  int totalElements = n_rows * n_columns;
  dim3 block(256);
  dim3 grid((totalElements + block.x - 1) / block.x );
  devLayoutIdx<OutType><<<grid, block, 0, stream>>>(in, n_columns, totalElements);
  return cudaGetLastError();
}

// helper function to layout offsets for rows for DeviceSegmentedRadixSort
template <typename T>
cudaError_t layoutSortOffset(T *in, T value, int n_times, cudaStream_t stream) {
  dim3 block(128);
  dim3 grid((n_times + block.x - 1) / block.x);
  devOffsetKernel<T><<<grid, block, 0, stream>>>(in, value, n_times);
  return cudaGetLastError();
}

/**
 * @brief sort columns within each row of row-major input matrix and return sorted indexes
 * modelled as key-value sort with key being input matrix and value being index of values 
 * @param in: input matrix
 * @param out: output value(index) matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param bAllocWorkspace: check returned value, if true allocate workspace passed in workspaceSize
 * @param workspacePtr: pointer to workspace memory
 * @param workspaceSize: Size of workspace to be allocated
 * @param stream: cuda stream to execute prim on
 * @param sortedKeys: Optional, output matrix for sorted keys (input)
 */
template <typename InType, typename OutType>
void sortColumnsPerRow(const InType *in, OutType *out, int n_rows, int n_columns,
                        bool &bAllocWorkspace, void *workspacePtr,
                        size_t &workspaceSize, cudaStream_t stream,
                        InType *sortedKeys=nullptr) {
  
  // assume non-square row-major matrices
  // current use-case: KNN, trustworthiness scores
  // output : either sorted indices or sorted indices and input values
  // future : this prim can be modified to be more generic and serve as a way to sort column entries per row
  //          i.e. another output format: sorted values only

  int totalElements = n_rows * n_columns;
  size_t perElementSmemUsage = sizeof(InType) + sizeof(OutType);
  size_t memAlignWidth = 256;

  // @ToDo: Figure out dynamic shared memory for block sort kernel - better for volta and beyond
  // int currDevice = 0, smemLimit = 0;
  // CUDA_CHECK(cudaGetDevice(&currDevice));
  // CUDA_CHECK(cudaDeviceGetAttribute(&smemLimit, cudaDevAttrMaxSharedMemoryPerBlock, currDevice));
  // size_t maxElementsForBlockSort = smemLimit / perElementSmemUsage;

  // for 48KB smem/block, can fit in 6144 4byte key-value pair
  // assuming key-value sort for now - smem computation will change for value only sort
  // dtype being size of key-value pair
  std::map<size_t, int> dtypeToColumnMap = { {4, 12288}, // short + short
                                          {8, 12288},  // float/int + int/float
                                          {12, 6144}, // double + int/float
                                          {16, 6144}}; // double + double

  if (dtypeToColumnMap.count(perElementSmemUsage) != 0 && 
        n_columns <= dtypeToColumnMap[perElementSmemUsage]) {

    // more elements per thread --> more register pressure
    // 512(blockSize) * 8 elements per thread = 71 register / thread

    // instantiate some kernel combinations
    if (n_columns <= 512)
      INST_BLOCK_SORT(in, sortedKeys, out, n_rows, n_columns, 128, 4, stream);
    else if(n_columns > 512 && n_columns <= 1024)
      INST_BLOCK_SORT(in, sortedKeys, out, n_rows, n_columns, 128, 8, stream);
    else if(n_columns > 1024 && n_columns <= 3072)
      INST_BLOCK_SORT(in, sortedKeys, out, n_rows, n_columns, 512, 6, stream);
    else if(n_columns > 3072 && n_columns <= 4096)
      INST_BLOCK_SORT(in, sortedKeys, out, n_rows, n_columns, 512, 8, stream);
    else if(n_columns > 4096 && n_columns <= 6144)
      INST_BLOCK_SORT(in, sortedKeys, out, n_rows, n_columns, 512, 12, stream);
    else
      INST_BLOCK_SORT(in, sortedKeys, out, n_rows, n_columns, 1024, 12, stream);
  }
  else if (n_columns <= (1 << 18) && n_rows > 1) {
      // device Segmented radix sort
      // 2^18 column cap to restrict size of workspace ~512 MB
      // will give better perf than below deviceWide Sort for even larger dims
      int numSegments = n_rows + 1;

      // need auxillary storage: cub sorting + keys (if user not passing) + 
      // staging for values out + segment partition
      if (workspaceSize == 0 || !workspacePtr) {
        OutType *tmpValIn = nullptr;
        int *tmpOffsetBuffer = nullptr;

        // first call is to get size of workspace
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(workspacePtr, workspaceSize, in, sortedKeys, 
                                                  tmpValIn, out, totalElements, numSegments, tmpOffsetBuffer, 
                                                  tmpOffsetBuffer+1));
        bAllocWorkspace = true;
        // more staging space for temp output of keys
        if (!sortedKeys)
          workspaceSize += alignTo(sizeof(InType) * (size_t)totalElements, memAlignWidth);
        
        // value in KV pair need to be passed in, out buffer is separate
        workspaceSize += alignTo(sizeof(OutType) * (size_t)totalElements, memAlignWidth);

        // for segment offsets
        workspaceSize += alignTo(sizeof(int) * (size_t)numSegments, memAlignWidth);
      }
      else {
          
          size_t workspaceOffset = 0;

          if (!sortedKeys) {
            sortedKeys = reinterpret_cast<InType *>(workspacePtr);
            workspaceOffset = alignTo(sizeof(InType) * (size_t)totalElements, memAlignWidth);
            workspacePtr = (void *)((size_t)workspacePtr + workspaceOffset);
          }

          OutType *dValuesIn = reinterpret_cast<OutType *>(workspacePtr);
          workspaceOffset = alignTo(sizeof(OutType) * (size_t)totalElements, memAlignWidth);
          workspacePtr = (void *)((size_t)workspacePtr + workspaceOffset);

          int *dSegmentOffsets = reinterpret_cast<int *>(workspacePtr);
          workspaceOffset = alignTo(sizeof(int) * (size_t)numSegments, memAlignWidth);
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
      if (workspaceSize == 0 || !workspacePtr) {
        OutType *tmpValIn = nullptr;

        // first call is to get size of workspace
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(workspacePtr, workspaceSize, in, sortedKeys, 
                                                  tmpValIn, out, n_columns));
        bAllocWorkspace = true;

        if (!sortedKeys)
          workspaceSize += alignTo(sizeof(InType) * (size_t)n_columns, memAlignWidth);

        workspaceSize += alignTo(sizeof(OutType) * (size_t)n_columns, memAlignWidth);
      }
      else {

        size_t workspaceOffset = 0;
        bool userKeyOutputBuffer = true;

        if (!sortedKeys) {
          userKeyOutputBuffer = false;
          sortedKeys = reinterpret_cast<InType *>(workspacePtr);
          workspaceOffset = alignTo(sizeof(InType) * (size_t)n_columns, memAlignWidth);
          workspacePtr = (void *)((size_t)workspacePtr + workspaceOffset);
        }

        OutType *dValuesIn = reinterpret_cast<OutType *>(workspacePtr);
        workspaceOffset = alignTo(sizeof(OutType) * (size_t)n_columns, memAlignWidth);
        workspacePtr = (void *)((size_t)workspacePtr + workspaceOffset);

        // layout idx
        CUDA_CHECK(layoutIdx(dValuesIn, 1, n_columns, stream));

        for (int i = 0; i < n_rows; i++) {
          InType *rowIn = reinterpret_cast<InType*>((size_t)in + (i * sizeof(InType) * (size_t)n_columns));
          OutType *rowOut = reinterpret_cast<OutType*>((size_t)out + (i * sizeof(OutType) * (size_t)n_columns));

          CUDA_CHECK(cub::DeviceRadixSort::SortPairs(workspacePtr, workspaceSize, rowIn, sortedKeys,
                                                  dValuesIn, rowOut, n_columns));

          if (userKeyOutputBuffer)
            sortedKeys = reinterpret_cast<InType *>((size_t)sortedKeys + sizeof(InType) * (size_t)n_columns);
        }
      }
  }
}
}; // end namespace Selection
}; // end namespace MLCommon
