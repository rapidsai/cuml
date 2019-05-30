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
/**
 * @file randIndex.h
 * @todo TODO(Ganesh Venkataramana):
 * <pre>
 * The below randIndex calculation implementation is a Brute force one that uses (nElements*nElements) threads (2 dimensional grids and blocks)
 * For small datasets, this will suffice; but for larger ones, work done by the threads increase dramatically. 
 * A more mathematically intensive implementation that uses half the above threads can be done, which will prove to be more efficient for larger datasets
 * the idea is as follows:
  * instead of 2D block and grid configuration with a total of (nElements*nElements) threads (where each (i,j) through these threads represent an ordered pair selection of 2 data points),
  a 1D block and grid configuration with a total of (nElements*(nElements))/2 threads (each thread index represents an element part of the set of unordered pairwise selections from the dataset (nChoose2))
  * In this setup, one has to generate a one-to-one mapping between this 1D thread index (for each kernel) and the unordered pair of chosen datapoints.
  * More specifically, thread0-> {dataPoint1, dataPoint0}, thread1-> {dataPoint2, dataPoint0}, thread2-> {dataPoint2, dataPoint1} ... thread((nElements*(nElements))/2 - 1)-> {dataPoint(nElements-1),dataPoint(nElements-2)}
  * say ,
     * threadNum: thread index | threadNum = threadIdx.x + BlockIdx.x*BlockDim.x,
     * i : index of dataPoint i
     * j : index of dataPoint j
  * then the mapping is as follows:
     * i = ceil((-1 + sqrt(1 + 8*(1 + threadNum)))/2) = floor((1 + sqrt(1 + 8*threadNum))/2)
     * j = threadNum - i(i-1)/2
  * after obtaining the the pair of datapoints, calculation of rand index is the same as done in this implementation
 * Caveat: since the kernel implementation involves use of emulated sqrt() operations:
  * the number of instructions executed per kernel is ~40-50 times
  * as the O(nElements*nElements) increase beyond the floating point limit, floating point inaccuracies occur, and hence the above floor(...) !=  ceil(...)
 * </pre>
 */
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include <cub/cub.cuh>
#include "cuda_utils.h"
#include <math.h>

namespace MLCommon {
namespace Metrics {

/**
 * @brief kernel to calculate the values of a and b 
 * @param firstClusterArray: the array of classes of type T 
 * @param secondClusterArray: the array of classes of type T
 * @param size: the size of the data points
 * @param a: number of pairs of points that both the clusters have classified the same
 * @param b: number of pairs of points that both the clusters have classified differently
 */
template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ 
void computeTheNumerator(const T* firstClusterArray, const T* secondClusterArray, uint64_t size, uint64_t *a, uint64_t *b) {

  //calculating the indices of pairs of datapoints compared by the current thread
  uint64_t j = threadIdx.x + blockIdx.x*blockDim.x;
  uint64_t i = threadIdx.y + blockIdx.y*blockDim.y;

  //thread-local variables to count a and b
  uint64_t myA = 0, myB = 0;

  if(i < size && j < size && j<i) {

  //checking if the pair have been classified the same by both the clusters
  if(firstClusterArray[i]==firstClusterArray[j]&&secondClusterArray[i]==secondClusterArray[j]){
    ++myA;
  }

  //checking if the pair have been classified differently by both the clusters
  else if(firstClusterArray[i]!=firstClusterArray[j]&&secondClusterArray[i]!=secondClusterArray[j]){
    ++myB;
    }
  }

  //specialize blockReduce for a 2D block of 1024 threads of type uint64_t
  typedef cub::BlockReduce <uint64_t, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y> BlockReduce;

  //Allocate shared memory for blockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;
  
  //summing up thread-local counts specific to a block
  myA = BlockReduce(temp_storage).Sum(myA);
  __syncthreads();
  myB = BlockReduce(temp_storage).Sum(myB);
  __syncthreads();
  
  //executed once per block
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd((unsigned long long int*)a, myA);
    atomicAdd((unsigned long long int*)b, myB);
  }
}


/**
* @brief Function to calculate RandIndex
* @param firstClusterArray: the array of classes of type T
* @param secondClusterArray: the array of classes of type T
* @param size: the size of the data points of type uint64_t
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename T>
float computeRandIndex (T* firstClusterArray, T* secondClusterArray, uint64_t size,
                       std::shared_ptr<MLCommon::deviceAllocator> allocator, cudaStream_t stream) {

  //allocating and initializing memory for a and b in the GPU
  MLCommon::device_buffer<uint64_t> arr_buf (allocator, stream, 2);
  CUDA_CHECK(cudaMemsetAsync(arr_buf.data(),0,2*sizeof(uint64_t),stream));

  //kernel configuration
  static const int BLOCK_DIM_Y = 16, BLOCK_DIM_X = 16;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(size,numThreadsPerBlock.x),ceildiv<int>(size,numThreadsPerBlock.y));

  //calling the kernel
  computeTheNumerator<T, BLOCK_DIM_X, BLOCK_DIM_Y><<<numBlocks,numThreadsPerBlock,0,stream>>>(firstClusterArray,secondClusterArray,size,arr_buf.data(),arr_buf.data()+1);

  //synchronizing and updating the calculated values of a and b from device to host
  uint64_t ab_host[2] = {0};
  MLCommon::updateHost(ab_host, arr_buf.data(), 2, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  //error handling
  CUDA_CHECK(cudaGetLastError());

  //denominator
  uint64_t nChooseTwo = size*(size-1)/2;

  //calculating the randIndex
  return (float)(((float)(ab_host[0] + ab_host[1]))/(float)nChooseTwo);
}


};//end namespace Metrics
};//end namespace MLCommon

