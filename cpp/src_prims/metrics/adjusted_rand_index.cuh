/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
 * @file adjusted_rand_index.cuh
 * @brief The adjusted Rand index is the corrected-for-chance version of the Rand index.
 * Such a correction for chance establishes a baseline by using the expected similarity
 * of all pair-wise comparisons between clusterings specified by a random model.
 */

#pragma once

#include "contingencyMatrix.cuh"
#include <cub/cub.cuh>
#include <math.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/linalg/map_then_reduce.hpp>
#include <raft/linalg/reduce.hpp>
#include <raft/stats/histogram.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

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
  HDI Type operator()(Type in, int i = 0)
  {
    return in % 2 ? ((in - 1) >> 1) * in : (in >> 1) * (in - 1);
  }
};

template <typename DataT, typename IdxT>
struct Binner {
  Binner(DataT minL) : minLabel(minL) {}

  DI int operator()(DataT val, IdxT row, IdxT col) { return int(val - minLabel); }

 private:
  DataT minLabel;
};  // struct Binner

/**
 * @brief Function to count the number of unique elements in the input array
 *
 * @tparam T data-type for input arrays
 *
 * @param[in]  arr       input array [on device] [len = size]
 * @param[in]  size      the size of the input array
 * @param[out] minLabel  the lower bound of the range of labels
 * @param[out] maxLabel  the upper bound of the range of labels
 * @param[in]  stream    cuda stream
 *
 * @return the number of unique elements in the array
 */
template <typename T>
int countUnique(const T* arr, int size, T& minLabel, T& maxLabel, cudaStream_t stream)
{
  auto ptr         = thrust::device_pointer_cast(arr);
  auto minmax      = thrust::minmax_element(thrust::cuda::par.on(stream), ptr, ptr + size);
  minLabel         = *minmax.first;
  maxLabel         = *minmax.second;
  auto totalLabels = int(maxLabel - minLabel + 1);
  rmm::device_uvector<int> labelCounts(totalLabels, stream);
  rmm::device_scalar<int> nUniq(stream);
  raft::stats::histogram<T, int>(
    raft::stats::HistTypeAuto,
    labelCounts.data(),
    totalLabels,
    arr,
    size,
    1,
    stream,
    [minLabel] __device__(T val, int row, int col) { return int(val - minLabel); });
  raft::linalg::mapThenSumReduce<int>(
    nUniq.data(),
    totalLabels,
    [] __device__(const T& val) { return val != 0; },
    stream,
    labelCounts.data());
  auto numUniques = nUniq.value(stream);
  return numUniques;
}

/**
 * @brief Function to calculate Adjusted RandIndex as described
 *        <a href="https://en.wikipedia.org/wiki/Rand_index">here</a>
 * @tparam T data-type for input label arrays
 * @tparam MathT integral data-type used for computing n-choose-r
 * @param firstClusterArray: the array of classes
 * @param secondClusterArray: the array of classes
 * @param size: the size of the data points of type int
 * @param stream: the cudaStream object
 */
template <typename T, typename MathT = int>
double compute_adjusted_rand_index(const T* firstClusterArray,
                                   const T* secondClusterArray,
                                   int size,
                                   cudaStream_t stream)
{
  ASSERT(size >= 2, "Rand Index for size less than 2 not defined!");
  T minFirst, maxFirst, minSecond, maxSecond;
  auto nUniqFirst      = countUnique(firstClusterArray, size, minFirst, maxFirst, stream);
  auto nUniqSecond     = countUnique(secondClusterArray, size, minSecond, maxSecond, stream);
  auto lowerLabelRange = std::min(minFirst, minSecond);
  auto upperLabelRange = std::max(maxFirst, maxSecond);
  auto nClasses        = upperLabelRange - lowerLabelRange + 1;
  // degenerate case of single cluster or clusters each with just one element
  if (nUniqFirst == nUniqSecond) {
    if (nUniqFirst == 1 || nUniqFirst == size) return 1.0;
  }
  auto nUniqClasses = MathT(nClasses);
  rmm::device_uvector<MathT> dContingencyMatrix(nUniqClasses * nUniqClasses, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(
    dContingencyMatrix.data(), 0, nUniqClasses * nUniqClasses * sizeof(MathT), stream));
  auto workspaceSz = getContingencyMatrixWorkspaceSize<T, MathT>(
    size, firstClusterArray, stream, lowerLabelRange, upperLabelRange);
  rmm::device_uvector<char> workspaceBuff(workspaceSz, stream);
  contingencyMatrix<T, MathT>(firstClusterArray,
                              secondClusterArray,
                              size,
                              dContingencyMatrix.data(),
                              stream,
                              workspaceBuff.data(),
                              workspaceSz,
                              lowerLabelRange,
                              upperLabelRange);
  rmm::device_uvector<MathT> a(nUniqClasses, stream);
  rmm::device_uvector<MathT> b(nUniqClasses, stream);
  rmm::device_scalar<MathT> d_aCTwoSum(stream);
  rmm::device_scalar<MathT> d_bCTwoSum(stream);
  rmm::device_scalar<MathT> d_nChooseTwoSum(stream);
  MathT h_aCTwoSum, h_bCTwoSum, h_nChooseTwoSum;
  RAFT_CUDA_TRY(cudaMemsetAsync(a.data(), 0, nUniqClasses * sizeof(MathT), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(b.data(), 0, nUniqClasses * sizeof(MathT), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_aCTwoSum.data(), 0, sizeof(MathT), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_bCTwoSum.data(), 0, sizeof(MathT), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_nChooseTwoSum.data(), 0, sizeof(MathT), stream));
  // calculating the sum of NijC2
  raft::linalg::mapThenSumReduce<MathT, nCTwo<MathT>>(d_nChooseTwoSum.data(),
                                                      nUniqClasses * nUniqClasses,
                                                      nCTwo<MathT>(),
                                                      stream,
                                                      dContingencyMatrix.data(),
                                                      dContingencyMatrix.data());
  // calculating the row-wise sums
  raft::linalg::reduce<MathT, MathT>(
    a.data(), dContingencyMatrix.data(), nUniqClasses, nUniqClasses, 0, true, true, stream);
  // calculating the column-wise sums
  raft::linalg::reduce<MathT, MathT>(
    b.data(), dContingencyMatrix.data(), nUniqClasses, nUniqClasses, 0, true, false, stream);
  // calculating the sum of number of unordered pairs for every element in a
  raft::linalg::mapThenSumReduce<MathT, nCTwo<MathT>>(
    d_aCTwoSum.data(), nUniqClasses, nCTwo<MathT>(), stream, a.data(), a.data());
  // calculating the sum of number of unordered pairs for every element of b
  raft::linalg::mapThenSumReduce<MathT, nCTwo<MathT>>(
    d_bCTwoSum.data(), nUniqClasses, nCTwo<MathT>(), stream, b.data(), b.data());
  // updating in the host memory
  raft::update_host(&h_nChooseTwoSum, d_nChooseTwoSum.data(), 1, stream);
  raft::update_host(&h_aCTwoSum, d_aCTwoSum.data(), 1, stream);
  raft::update_host(&h_bCTwoSum, d_bCTwoSum.data(), 1, stream);
  // calculating the ARI
  auto nChooseTwo    = double(size) * double(size - 1) / 2.0;
  auto expectedIndex = double(h_aCTwoSum) * double(h_bCTwoSum) / double(nChooseTwo);
  auto maxIndex      = (double(h_bCTwoSum) + double(h_aCTwoSum)) / 2.0;
  auto index         = double(h_nChooseTwoSum);
  if (maxIndex - expectedIndex)
    return (index - expectedIndex) / (maxIndex - expectedIndex);
  else
    return 0;
}

};  // end namespace Metrics
};  // end namespace MLCommon
