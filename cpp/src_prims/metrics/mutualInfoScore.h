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
* @file mutualInfoScore.h
* @brief The Mutual Information is a measure of the similarity between two labels of
*   the same data.This metric is independent of the absolute values of the labels:
*   a permutation of the class or cluster label values won't change the
*   score value in any way.
*   This metric is furthermore symmetric.This can be useful to
*   measure the agreement of two independent label assignments strategies
*   on the same dataset when the real ground truth is not known.
*/

#include <math.h>
#include <cub/cub.cuh>
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "linalg/reduce.h"
#include "metrics/contingencyMatrix.h"

namespace MLCommon {

namespace Metrics {

/**
 * @brief kernel to calculate the mutual info score
 * @param dContingencyMatrix: the contingency matrix corresponding to the two clusters
 * @param a: the row wise sum of the contingency matrix, which is also the bin counts of first cluster array
 * @param b: the column wise sum of the contingency matrix, which is also the bin counts of second cluster array
 * @param size: the size of array a and b (size of the contingency matrix is (size x size))
 * @param d_MI: pointer to the device memory that stores the aggreggate mutual information
 */
template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void mutualInfoKernel(const int *dContingencyMatrix, const int *a,
                                 const int *b, int numUniqueClasses, int size,
                                 double *d_MI) {
  //calculating the indices of pairs of datapoints compared by the current thread
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  //thread-local variable to count the mutual info
  double localMI = 0.0;

  if (i < numUniqueClasses && j < numUniqueClasses && a[i] * b[j] != 0 &&
      dContingencyMatrix[i * numUniqueClasses + j] != 0) {
    localMI += (double(dContingencyMatrix[i * numUniqueClasses + j])) *
               (log(double(size) *
                    double(dContingencyMatrix[i * numUniqueClasses + j])) -
                log(double(a[i] * b[j])));
  }

  //specialize blockReduce for a 2D block of 1024 threads of type uint64_t
  typedef cub::BlockReduce<double, BLOCK_DIM_X,
                           cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y>
    BlockReduce;

  //Allocate shared memory for blockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;

  //summing up thread-local counts specific to a block
  localMI = BlockReduce(temp_storage).Sum(localMI);
  __syncthreads();

  //executed once per block
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    myAtomicAdd(d_MI, localMI);
  }
}

/**
* @brief Function to calculate the mutual information between two clusters
* <a href="https://en.wikipedia.org/wiki/Mutual_information">more info on mutual information</a> 
* @param firstClusterArray: the array of classes of type T
* @param secondClusterArray: the array of classes of type T
* @param size: the size of the data points of type int
* @param numUniqueClasses: number of Unique classes used for clustering
* @param lowerLabelRange: the lower bound of the range of labels
* @param upperLabelRange: the upper bound of the range of labels
* @param allocator: object that takes care of temporary device memory allocation of type std::shared_ptr<MLCommon::deviceAllocator>
* @param stream: the cudaStream object
*/
template <typename T>
double mutualInfoScore(const T *firstClusterArray, const T *secondClusterArray,
                       int size, T lowerLabelRange, T upperLabelRange,
                       std::shared_ptr<MLCommon::deviceAllocator> allocator,
                       cudaStream_t stream) {
  int numUniqueClasses = upperLabelRange - lowerLabelRange + 1;

  //declaring, allocating and initializing memory for the contingency marix
  MLCommon::device_buffer<int> dContingencyMatrix(
    allocator, stream, numUniqueClasses * numUniqueClasses);
  CUDA_CHECK(cudaMemsetAsync(dContingencyMatrix.data(), 0,
                             numUniqueClasses * numUniqueClasses * sizeof(int),
                             stream));

  //workspace allocation
  size_t workspaceSz = MLCommon::Metrics::getContingencyMatrixWorkspaceSize(
    size, firstClusterArray, stream, lowerLabelRange, upperLabelRange);
  device_buffer<char> pWorkspace(allocator, stream, workspaceSz);

  //calculating the contingency matrix
  MLCommon::Metrics::contingencyMatrix(
    firstClusterArray, secondClusterArray, (int)size,
    (int *)dContingencyMatrix.data(), stream, (void *)pWorkspace.data(),
    workspaceSz, lowerLabelRange, upperLabelRange);

  //creating device buffers for all the parameters involved in ARI calculation
  //device variables
  MLCommon::device_buffer<int> a(allocator, stream, numUniqueClasses);
  MLCommon::device_buffer<int> b(allocator, stream, numUniqueClasses);
  MLCommon::device_buffer<double> d_MI(allocator, stream, 1);

  //host variables
  double h_MI;

  //initializing device memory
  CUDA_CHECK(
    cudaMemsetAsync(a.data(), 0, numUniqueClasses * sizeof(int), stream));
  CUDA_CHECK(
    cudaMemsetAsync(b.data(), 0, numUniqueClasses * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_MI.data(), 0, sizeof(double), stream));

  //calculating the row-wise sums
  MLCommon::LinAlg::reduce<int, int, int>(a.data(), dContingencyMatrix.data(),
                                          numUniqueClasses, numUniqueClasses, 0,
                                          true, true, stream);

  //calculating the column-wise sums
  MLCommon::LinAlg::reduce<int, int, int>(b.data(), dContingencyMatrix.data(),
                                          numUniqueClasses, numUniqueClasses, 0,
                                          true, false, stream);

  //kernel configuration
  static const int BLOCK_DIM_Y = 16, BLOCK_DIM_X = 16;
  dim3 numThreadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 numBlocks(ceildiv<int>(size, numThreadsPerBlock.x),
                 ceildiv<int>(size, numThreadsPerBlock.y));

  //calling the kernel
  mutualInfoKernel<T, BLOCK_DIM_X, BLOCK_DIM_Y>
    <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      dContingencyMatrix.data(), a.data(), b.data(), numUniqueClasses, size,
      d_MI.data());

  //updating in the host memory
  MLCommon::updateHost(&h_MI, d_MI.data(), 1, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return h_MI / size;
}

};  //end namespace Metrics
};  //end namespace MLCommon
