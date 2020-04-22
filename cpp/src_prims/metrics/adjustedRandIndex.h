/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <math.h>
#include <cub/cub.cuh>
#include <cuml/common/cuml_allocator.hpp>
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "linalg/map_then_reduce.h"
#include "linalg/reduce.h"
#include "metrics/contingencyMatrix.h"

namespace MLCommon {
namespace Metrics {

/**
 * @brief Lambda to calculate the number of unordered pairs in a given input
 *
 * @tparam Type: Data type of the input 
 * @param in: the input to the functional mapping
 * @param i: the indexing(not used in this case)
 */
template <typename Type>
struct nCTwo {
  HDI Type operator()(Type in, int i = 0) {
    return in % 2 ? ((in - 1) >> 1) * in : (in >> 1) * (in - 1);
  }
};

/**
* @brief Function to calculate Adjusted RandIndex as described
*        <a href="https://en.wikipedia.org/wiki/Rand_index">here</a>
* @tparam T data-type for input label arrays
* @tparam MathT integral data-type used for computing n-choose-r
* @param firstClusterArray: the array of classes
* @param secondClusterArray: the array of classes
* @param size: the size of the data points of type int
* @param numUniqueClasses: number of Unique classes used for clustering
* @param lowerLabelRange: the lower bound of the range of labels
* @param upperLabelRange: the upper bound of the range of labels
* @param allocator: object that takes care of temporary device memory allocation
* @param stream: the cudaStream object
*/
template <typename T, typename MathT = int>
double computeAdjustedRandIndex(const T* firstClusterArray,
                                const T* secondClusterArray, int size,
                                T lowerLabelRange, T upperLabelRange,
                                std::shared_ptr<deviceAllocator> allocator,
                                cudaStream_t stream) {
  ASSERT(size >= 2, "Rand Index for size less than 2 not defined!");
  auto nUniqClasses = MathT(upperLabelRange - lowerLabelRange + 1);
  device_buffer<MathT> dContingencyMatrix(allocator, stream,
                                          nUniqClasses * nUniqClasses);
  CUDA_CHECK(cudaMemsetAsync(dContingencyMatrix.data(), 0,
                             nUniqClasses * nUniqClasses * sizeof(MathT),
                             stream));
  auto workspaceSz = getContingencyMatrixWorkspaceSize<T, MathT>(
    size, firstClusterArray, stream, lowerLabelRange, upperLabelRange);
  device_buffer<char> workspaceBuff(allocator, stream, workspaceSz);
  contingencyMatrix<T, MathT>(firstClusterArray, secondClusterArray, size,
                              dContingencyMatrix.data(), stream,
                              workspaceBuff.data(), workspaceSz,
                              lowerLabelRange, upperLabelRange);
  device_buffer<MathT> a(allocator, stream, nUniqClasses);
  device_buffer<MathT> b(allocator, stream, nUniqClasses);
  device_buffer<MathT> d_aCTwoSum(allocator, stream, 1);
  device_buffer<MathT> d_bCTwoSum(allocator, stream, 1);
  device_buffer<MathT> d_nChooseTwoSum(allocator, stream, 1);
  MathT h_aCTwoSum, h_bCTwoSum, h_nChooseTwoSum;
  CUDA_CHECK(
    cudaMemsetAsync(a.data(), 0, nUniqClasses * sizeof(MathT), stream));
  CUDA_CHECK(
    cudaMemsetAsync(b.data(), 0, nUniqClasses * sizeof(MathT), stream));
  CUDA_CHECK(cudaMemsetAsync(d_aCTwoSum.data(), 0, sizeof(MathT), stream));
  CUDA_CHECK(cudaMemsetAsync(d_bCTwoSum.data(), 0, sizeof(MathT), stream));
  CUDA_CHECK(cudaMemsetAsync(d_nChooseTwoSum.data(), 0, sizeof(MathT), stream));
  //calculating the sum of NijC2
  LinAlg::mapThenSumReduce<MathT, nCTwo<MathT>>(
    d_nChooseTwoSum.data(), nUniqClasses * nUniqClasses, nCTwo<MathT>(), stream,
    dContingencyMatrix.data(), dContingencyMatrix.data());
  //calculating the row-wise sums
  LinAlg::reduce<MathT, MathT>(a.data(), dContingencyMatrix.data(),
                               nUniqClasses, nUniqClasses, 0, true, true,
                               stream);
  //calculating the column-wise sums
  LinAlg::reduce<MathT, MathT>(b.data(), dContingencyMatrix.data(),
                               nUniqClasses, nUniqClasses, 0, true, false,
                               stream);
  //calculating the sum of number of unordered pairs for every element in a
  LinAlg::mapThenSumReduce<MathT, nCTwo<MathT>>(d_aCTwoSum.data(), nUniqClasses,
                                                nCTwo<MathT>(), stream,
                                                a.data(), a.data());
  //calculating the sum of number of unordered pairs for every element of b
  LinAlg::mapThenSumReduce<MathT, nCTwo<MathT>>(d_bCTwoSum.data(), nUniqClasses,
                                                nCTwo<MathT>(), stream,
                                                b.data(), b.data());
  //updating in the host memory
  updateHost(&h_nChooseTwoSum, d_nChooseTwoSum.data(), 1, stream);
  updateHost(&h_aCTwoSum, d_aCTwoSum.data(), 1, stream);
  updateHost(&h_bCTwoSum, d_bCTwoSum.data(), 1, stream);
  //calculating the ARI
  auto nChooseTwo = double(size) * double(size - 1) / 2.0;
  auto expectedIndex =
    double(h_aCTwoSum) * double(h_bCTwoSum) / double(nChooseTwo);
  auto maxIndex = (double(h_bCTwoSum) + double(h_aCTwoSum)) / 2.0;
  auto index = double(h_nChooseTwoSum);
  if (maxIndex - expectedIndex)
    return (index - expectedIndex) / (maxIndex - expectedIndex);
  else
    return 0;
}

};  //end namespace Metrics
};  //end namespace MLCommon
