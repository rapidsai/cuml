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

#include "cuda_utils.h"
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <cub/cub.cuh>
#include <math.h>

#define ALIGN_BYTE 256
#define ALIGN_MEMORY(x) (x + ALIGN_BYTE - 1) & ~(ALIGN_BYTE - 1)

namespace MLCommon {
namespace Metrics {

typedef enum {
  IMPL_NONE,
  SMEM_ATOMICS,
  GLOBAL_ATOMICS,
  SORT_AND_GATOMICS
} ContingencyMatrixImplType;

template <typename T>
__global__ void devConstructContingencyMatrix(T *groundTruth, T *predicted,
                                                int nSamples, int *outMat,
                                                int outIdxOffset, int outMatWidth) {
  int elementId = threadIdx.x + blockDim.x * blockIdx.x;
  if (elementId < nSamples) {
    T gt = groundTruth[elementId];
    T pd = predicted[elementId];
    
    int outputIdx = (gt - outIdxOffset) * outMatWidth + pd - outIdxOffset;
    myAtomicAdd(&outMat[outputIdx], 1);
  }
}

template <typename T>
__global__ void devConstructContingencyMatrixSmem(T *groundTruth, T *predicted,
                                                int nSamples, int *outMat,
                                                int outIdxOffset, int outMatWidth) {
  
  extern __shared__ int sMemMatrix[];  // init smem to zero
  
  // get linear smem ids form threadIdx's to smem to set to zero
  // set to zero
  for (int smemIdx=threadIdx.x; smemIdx < outMatWidth*outMatWidth; smemIdx+=blockDim.x) {
    sMemMatrix[smemIdx] = 0;
  }
  __syncthreads();

  int elementId = threadIdx.x + blockDim.x * blockIdx.x;
  if (elementId < nSamples) {
    T gt = groundTruth[elementId];
    T pd = predicted[elementId];
    
    int outputIdx = (gt - outIdxOffset) * outMatWidth + pd - outIdxOffset;
    myAtomicAdd(&sMemMatrix[outputIdx], 1);
  }
  __syncthreads();

  // upstream atomic updates to global matrix
  for (int smemIdx=threadIdx.x; smemIdx < outMatWidth*outMatWidth; smemIdx+=blockDim.x) {
    myAtomicAdd(&outMat[smemIdx], sMemMatrix[smemIdx]);
  }
}

// helper functions to launch kernel for global atomic add
template <typename T>
cudaError_t computeCMatWAtomics(T *groundTruth, T *predictedLabel, int nSamples,
                              int *outMat, int outIdxOffset, int outDimN,
                              cudaStream_t stream) {
  CUDA_CHECK(cudaFuncSetCacheConfig(devConstructContingencyMatrix<T>, cudaFuncCachePreferL1));                              
  dim3 block(128,1,1);
  dim3 grid((nSamples + block.x - 1) / block.x);

  // launch kernel - global atomic ops per groundTruth - predictedValue pair
  devConstructContingencyMatrix<<<grid, block, 0, stream>>>(groundTruth,
                              predictedLabel, nSamples, outMat, outIdxOffset, outDimN);

  return cudaGetLastError();
}

// helper function to launch share memory atomic add kernel
template <typename T>
cudaError_t computeCMatWSmemAtomics(T *groundTruth, T *predictedLabel, int nSamples,
                              int *outMat, int outIdxOffset, int outDimN,
                              cudaStream_t stream) {
  dim3 block(128,1,1);
  dim3 grid((nSamples + block.x - 1) / block.x);
  size_t smemSizePerBlock = outDimN * outDimN * sizeof(int);

  devConstructContingencyMatrixSmem<<<grid, block, smemSizePerBlock, stream>>>(groundTruth,
                              predictedLabel, nSamples, outMat, outIdxOffset, outDimN);

  return cudaGetLastError();
}

// helper function to sort and global atomic update
template <typename T>
void contingencyMatrixWSort(T *groundTruth, T *predictedLabel, int nSamples,
                        int *outMat, T minLabel, T maxLabel, void *workspace,
                        size_t workspaceSize, cudaStream_t stream) {

  T *outKeys = reinterpret_cast<T*>(workspace);
  size_t alignedBufferSz = ALIGN_MEMORY((size_t)nSamples * sizeof(T));
  T *outValue = reinterpret_cast<T*>((size_t)workspace + alignedBufferSz);

  void *pWorkspaceCub = reinterpret_cast<void *>((size_t)workspace + 2*alignedBufferSz);
  int bitsToSort = int(std::ceil(std::log2f((float)maxLabel)));


  // we dont really need perfect sorting, should get by with some sort of binning-reordering operation
  // future work - explore "efficient" custom binning kernels vs cub sort
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(pWorkspaceCub, workspaceSize, groundTruth, outKeys,
                                              predictedLabel, outValue, nSamples, 0,
                                              bitsToSort, stream));
  int outDimM_N = (int)(maxLabel - minLabel + T(1));

  computeCMatWAtomics(outKeys, outValue, nSamples, outMat, minLabel, outDimM_N, stream);
}

inline ContingencyMatrixImplType getImplVersion(int outDimN) {
  int currDevice = 0;
  int l2CacheSize = 0;
  int maxSmemPerBlock = 0;
  int maxBlocksResidentPerSM = 16;    // no way to query this from CUDA APIs, value for CC 7.0, 3.0

  CUDA_CHECK(cudaGetDevice(&currDevice));
  CUDA_CHECK(cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, currDevice));
  CUDA_CHECK(cudaDeviceGetAttribute(&maxSmemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, currDevice));

  ContingencyMatrixImplType implVersion = IMPL_NONE;

  // keeping 8 block per SM to get good utilization
  // can go higher but reduced L1 size degrades perf
  int upperLimitSmemAtomics = std::floor(std::sqrt(maxSmemPerBlock / (sizeof(int) *(maxBlocksResidentPerSM / 2))));
  int upperLimitL2Atomics = std::floor(std::sqrt(l2CacheSize / sizeof(int) ));

  if (outDimN <= upperLimitSmemAtomics)
    implVersion = SMEM_ATOMICS;
  else if(outDimN <= upperLimitL2Atomics)
    implVersion = GLOBAL_ATOMICS;
  else
    implVersion = SORT_AND_GATOMICS;

  return implVersion;
}

/**
 * @brief use this to allocate output matrix size
 * size of matrix = (maxLabel - minLabel + 1)^2 * sizeof(int)
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param nSamples: number of elements in input array
 * @param stream: cuda stream for execution
 * @param minLabel: [out] calculated min value in input array
 * @param maxLabel: [out] calculated max value in input array
*/
template <typename T>
void getInputClassCardinality(T* groundTruth, int nSamples, cudaStream_t stream, T &minLabel, T &maxLabel) {
  thrust::device_ptr<T> dTrueLabel = thrust::device_pointer_cast(groundTruth);
    auto min_max = thrust::minmax_element(thrust::cuda::par.on(stream),
                                            dTrueLabel, dTrueLabel + nSamples);
    minLabel = *min_max.first;
    maxLabel = *min_max.second;
}

/**
 * @brief Calculate workspace size for running contingency matrix calculations
 * @param nSamples: number of elements in input array
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param stream: cuda stream for execution
 * @param minLabel: Optional, min value in input array
 * @param maxLabel: Optional, max value in input array
 */
template <typename T>
size_t getCMatrixWorkspaceSize(int nSamples, T* groundTruth, cudaStream_t stream, 
                                T minLabel=std::numeric_limits<T>::max(),
                                T maxLabel=std::numeric_limits<T>::max()) {
  size_t workspaceSize = 0;
  // below is a redundant computation - can be avoided
  if (minLabel == std::numeric_limits<T>::max() || 
        maxLabel == std::numeric_limits<T>::max()) {
    thrust::device_ptr<T> dTrueLabel = thrust::device_pointer_cast(groundTruth);
    auto min_max = thrust::minmax_element(thrust::cuda::par.on(stream),
                                            dTrueLabel, dTrueLabel + nSamples);
    minLabel = *min_max.first;
    maxLabel = *min_max.second;
  }

  int outDimN = int(maxLabel - minLabel + T(1));
  ContingencyMatrixImplType implVersion = getImplVersion(outDimN);

  if (implVersion == SORT_AND_GATOMICS) {
    void *pWorkspaceCub = NULL;
    size_t tmpStorageBytes = 0;
    // bunch of no-op pointers to get workspace size
    T *pTmpKey, *pTmpValue, *pTmpKeyOut, *pTmpValueOut;

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(pWorkspaceCub, tmpStorageBytes, pTmpKey, pTmpValue,
                                                pTmpKeyOut, pTmpValueOut, nSamples));

    size_t tmpStagingMemorySize = ALIGN_MEMORY(nSamples * sizeof(T));
    tmpStagingMemorySize *= 2;
    workspaceSize = tmpStagingMemorySize + tmpStorageBytes;
  }
  return workspaceSize;
}

/**
 * @brief contruct contingency matrix given input ground truth and prediction labels
 * users should call function getInputClassCardinality to find and allocate memory for
 * output. Similarly workspace requirements should be checked using function getCMatrixWorkspaceSize
 * @param groundTruth: device 1-d array for ground truth (num of rows)
 * @param predictedLabel: device 1-d array for prediction (num of columns)
 * @param nSamples: number of elements in input array
 * @param outMat: output buffer for contingecy matrix
 * @param stream: cuda stream for execution
 * @param workspace: Optional, workspace memory allocation
 * @param workspaceSize: Optional, size of workspace memory
 * @param minLabel: Optional, min value in input ground truth array
 * @param maxLabel: Optional, max value in input ground truth array
 */
template <typename T>
void contingencyMatrix(T *groundTruth, T *predictedLabel, int nSamples, int *outMat,
                        cudaStream_t stream, void *workspace=NULL, size_t workspaceSize=0,
                        T minLabel=std::numeric_limits<T>::max(),
                        T maxLabel=std::numeric_limits<T>::max()) {
  // assumptions:
  // output is not at par with scikit learn - output will be square matrix always with 
  // numRows = numColumns = numOfClassesInTrueLabel
  // it is also assumed that true labels are monotically increasing with step count 1
  // if for some reason groundTruth completely skips some labels 
  // eg: {0,1,2,5} instead of {0,1,2,3} . Output matrix will still have empty rows for label value {3,4}

  // this also serves as way to measure co-occurence/joint counts for NLP tasks which
  // can be used to then compute pointwise mutual information and mutual information

  if (minLabel == std::numeric_limits<T>::max() || 
        maxLabel == std::numeric_limits<T>::max()) {
    thrust::device_ptr<T> dTrueLabel = thrust::device_pointer_cast(groundTruth);
    auto min_max = thrust::minmax_element(thrust::cuda::par.on(stream),
                                            dTrueLabel, dTrueLabel + nSamples);
    minLabel = *min_max.first;
    maxLabel = *min_max.second;
  }

  int outDimM_N = (int)(maxLabel - minLabel + T(1));

  //memset outMat to zero before atomic increments
  cudaMemsetAsync((void*)outMat, 0, sizeof(int) * outDimM_N * outDimM_N, stream);

  ContingencyMatrixImplType implVersion = getImplVersion(outDimM_N);

  switch (implVersion){
    case SMEM_ATOMICS:
      // smem atomics and then single global mem atomics only works
      // when all label count can fit in smem for a block
      // helps when GLOBAL_ATOMICS performance blocked by atomic update serialization
      // -when very less labels ~10 labels
      computeCMatWSmemAtomics(groundTruth, predictedLabel, nSamples, outMat, 
                                minLabel, outDimM_N, stream);
      break;
    case GLOBAL_ATOMICS:
      // launch kernel - global atomic ops per (groundTruth,predictedValue) pair
      computeCMatWAtomics(groundTruth, predictedLabel, nSamples, outMat, 
                            minLabel, outDimM_N, stream);
      break;
      // more L2 thrashing if atomic OPs land in completely different mem segment - when more labels
    case SORT_AND_GATOMICS:
      contingencyMatrixWSort(groundTruth, predictedLabel, nSamples,
                          outMat, minLabel, maxLabel, workspace,
                          workspaceSize, stream);
      break;
  }
}
};
};