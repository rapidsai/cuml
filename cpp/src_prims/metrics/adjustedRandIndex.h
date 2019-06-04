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
* @file adjustedRandIndex.h
* @brief The adjusted Rand index is the corrected-for-chance version of the Rand index.
* Such a correction for chance establishes a baseline by using the expected similarity 
* of all pair-wise comparisons between clusterings specified by a random model.
*/

#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include <cub/cub.cuh>
#include "cuda_utils.h"
#include <math.h>
#include "linalg/reduce.h"
#include "linalg/map_then_reduce.h"
#include "metrics/contingencyMatrix.h"

namespace MLCommon {

/**
* @brief Lambda to calculate the number of unordered pairs in a given input
*
* @tparam Type: Data type of the input 
* @tparam IdxType : type of the indexing (by default int)
* @param in: the input to the functional mapping
* @param i: the indexing(not used in this case)
*/
template <typename Type, typename IdxType = int>
struct nCTwo {
  HDI Type operator()(Type in, IdxType i = 0) { return ((in)*(in-1))/2; }
};



namespace Metrics {



/**
* @brief Function to calculate Adjusted RandIndex
* <a href="https://en.wikipedia.org/wiki/Rand_index">more info on rand index</a> 
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
double computeAdjustedRandIndex (T* firstClusterArray, T* secondClusterArray, int size, int numUniqueClasses, int lowerLabelRange, int upperLabelRange,
                       std::shared_ptr<MLCommon::deviceAllocator> allocator, cudaStream_t stream) {

  //rand index for size less than 2 is not defined
  ASSERT(size>=2,"Rand Index for size less than 2 not defined!");

  //declaring, allocating and initializing memory for the contingency marix
  MLCommon::device_buffer<int> dContingencyMatrix (allocator, stream, numUniqueClasses*numUniqueClasses);
  CUDA_CHECK(cudaMemsetAsync(dContingencyMatrix.data(), 0, numUniqueClasses*numUniqueClasses*sizeof(int), stream));

  //workspace allocation
  char *pWorkspace = nullptr;
  size_t workspaceSz = MLCommon::Metrics::getCMatrixWorkspaceSize(size, firstClusterArray,
                                                      stream, lowerLabelRange, upperLabelRange);    
  if (workspaceSz != 0)
      MLCommon::allocate(pWorkspace, workspaceSz);

  //calculating the contingency matrix
  MLCommon::Metrics::contingencyMatrix(firstClusterArray, secondClusterArray, (int)size, (int*)dContingencyMatrix.data(),
                                        stream, (void*)pWorkspace, workspaceSz, lowerLabelRange, upperLabelRange);

  //creating device buffers for all the parameters involved in ARI calculation
  //device variables
  MLCommon::device_buffer<int> a(allocator, stream, numUniqueClasses);
  MLCommon::device_buffer<int> b(allocator, stream, numUniqueClasses);
  MLCommon::device_buffer<int> d_aCTwoSum(allocator, stream, 1);
  MLCommon::device_buffer<int> d_bCTwoSum(allocator, stream, 1);
  MLCommon::device_buffer<int> d_nChooseTwoSum(allocator, stream, 1);
  //host variables
  int h_aCTwoSum;
  int h_bCTwoSum;
  int h_nChooseTwoSum;


  //initializing device memory
  CUDA_CHECK(cudaMemsetAsync(a.data(), 0, numUniqueClasses*sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(b.data(), 0, numUniqueClasses*sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_aCTwoSum.data(), 0, sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_bCTwoSum.data(), 0, sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_nChooseTwoSum.data(), 0, sizeof(int), stream));


  //calculating the sum of NijC2
  MLCommon::LinAlg::mapThenSumReduce<int, nCTwo<int> >(d_nChooseTwoSum.data(), numUniqueClasses*numUniqueClasses, nCTwo< int >(), stream, dContingencyMatrix.data(), dContingencyMatrix.data());

  //calculating the row-wise sums
  MLCommon::LinAlg::reduce<int,int, int>(a.data(), dContingencyMatrix.data(), numUniqueClasses, numUniqueClasses, 0, true, true, stream);

  //calculating the column-wise sums
  MLCommon::LinAlg::reduce<int,int, int>(b.data(), dContingencyMatrix.data(), numUniqueClasses, numUniqueClasses, 0, true, false, stream);

  //calculating the sum of number of unordered pairs for every element in a
  MLCommon::LinAlg::mapThenSumReduce<int, nCTwo<int>>(d_aCTwoSum.data(), numUniqueClasses, nCTwo< int >(), stream, a.data(), a.data());

  //calculating the sum of number of unordered pairs for every element of b
  MLCommon::LinAlg::mapThenSumReduce<int, nCTwo<int>>(d_bCTwoSum.data(), numUniqueClasses, nCTwo< int >(), stream, b.data(), b.data());

  //updating in the host memory
  MLCommon::updateHost(&h_nChooseTwoSum, d_nChooseTwoSum.data(), 1, stream);
  MLCommon::updateHost(&h_aCTwoSum, d_aCTwoSum.data(), 1, stream);
  MLCommon::updateHost(&h_bCTwoSum, d_bCTwoSum.data(), 1, stream);

  //freeing the memories in the device
  if (pWorkspace)
    CUDA_CHECK(cudaFree(pWorkspace));

  //calculating the ARI
  int nChooseTwo = ((size)*(size-1))/2;
  double expectedIndex = ( (double)( (h_aCTwoSum)*(h_bCTwoSum) ) )/( (double)(nChooseTwo) );
  double maxIndex = ((double)(h_bCTwoSum+h_aCTwoSum))/2.0;
  double index = (double)h_nChooseTwoSum;

  //checking if the denominator is zero
  if(maxIndex - expectedIndex)
    return (index - expectedIndex)/(maxIndex - expectedIndex);
  else return 0;

}


};//end namespace Metrics
};//end namespace MLCommon

