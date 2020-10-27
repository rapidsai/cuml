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

#include <raft/cudart_utils.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cuml/common/cuml_allocator.hpp>
#include <iostream>
#include <metrics/adjustedRandIndex.cuh>
#include <metrics/contingencyMatrix.cuh>
#include <random>
#include "test_utils.h"

namespace MLCommon {
namespace Metrics {

struct AdjustedRandIndexParam {
  int nElements;
  int lowerLabelRange;
  int upperLabelRange;
  bool sameArrays;
  double tolerance;
  // if this is true, then it is assumed that `sameArrays` is also true
  // further it also assumes `lowerLabelRange` and `upperLabelRange` are 0
  bool testZeroArray;
};

template <typename T, typename MathT = int>
class AdjustedRandIndexTest
  : public ::testing::TestWithParam<AdjustedRandIndexParam> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<AdjustedRandIndexParam>::GetParam();
    nElements = params.nElements;
    raft::allocate(firstClusterArray, nElements, true);
    raft::allocate(secondClusterArray, nElements, true);
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::shared_ptr<deviceAllocator> allocator(
      new raft::mr::device::default_allocator);
    if (!params.testZeroArray) {
      SetUpDifferentArrays();
    } else {
      SetupZeroArray();
    }
    //allocating and initializing memory to the GPU
    computedAdjustedRandIndex = computeAdjustedRandIndex<T, MathT>(
      firstClusterArray, secondClusterArray, nElements, allocator, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(firstClusterArray));
    CUDA_CHECK(cudaFree(secondClusterArray));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void SetUpDifferentArrays() {
    lowerLabelRange = params.lowerLabelRange;
    upperLabelRange = params.upperLabelRange;
    std::vector<int> arr1(nElements, 0);
    std::vector<int> arr2(nElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(lowerLabelRange,
                                                    upperLabelRange);
    std::generate(arr1.begin(), arr1.end(),
                  [&]() { return intGenerator(dre); });
    if (params.sameArrays) {
      arr2 = arr1;
    } else {
      std::generate(arr2.begin(), arr2.end(),
                    [&]() { return intGenerator(dre); });
    }
    // calculating golden output
    int numUniqueClasses = upperLabelRange - lowerLabelRange + 1;
    size_t sizeOfMat = numUniqueClasses * numUniqueClasses * sizeof(int);
    int *hGoldenOutput = (int *)malloc(sizeOfMat);
    memset(hGoldenOutput, 0, sizeOfMat);
    for (int i = 0; i < nElements; i++) {
      int row = arr1[i] - lowerLabelRange;
      int column = arr2[i] - lowerLabelRange;
      hGoldenOutput[row * numUniqueClasses + column] += 1;
    }
    int sumOfNijCTwo = 0;
    int *a = (int *)malloc(numUniqueClasses * sizeof(int));
    int *b = (int *)malloc(numUniqueClasses * sizeof(int));
    memset(a, 0, numUniqueClasses * sizeof(int));
    memset(b, 0, numUniqueClasses * sizeof(int));
    int sumOfAiCTwo = 0;
    int sumOfBiCTwo = 0;
    //calculating the sum of number of pairwise points in each index
    //and also the reducing contingency matrix along row and column
    for (int i = 0; i < numUniqueClasses; ++i) {
      for (int j = 0; j < numUniqueClasses; ++j) {
        int Nij = hGoldenOutput[i * numUniqueClasses + j];
        sumOfNijCTwo += ((Nij) * (Nij - 1)) / 2;
        a[i] += hGoldenOutput[i * numUniqueClasses + j];
        b[i] += hGoldenOutput[j * numUniqueClasses + i];
      }
    }
    //claculating the sum of number pairwise points in ever column sum
    //claculating the sum of number pairwise points in ever row sum
    for (int i = 0; i < numUniqueClasses; ++i) {
      sumOfAiCTwo += ((a[i]) * (a[i] - 1)) / 2;
      sumOfBiCTwo += ((b[i]) * (b[i] - 1)) / 2;
    }
    //calculating the ARI
    double nCTwo = double(nElements) * double(nElements - 1) / 2.0;
    double expectedIndex =
      (double(sumOfBiCTwo) * double(sumOfAiCTwo)) / double(nCTwo);
    double maxIndex = (double(sumOfAiCTwo) + double(sumOfBiCTwo)) / 2.0;
    double index = (double)sumOfNijCTwo;
    if (maxIndex - expectedIndex)
      truthAdjustedRandIndex =
        (index - expectedIndex) / (maxIndex - expectedIndex);
    else
      truthAdjustedRandIndex = 0;
    raft::update_device(firstClusterArray, &arr1[0], nElements, stream);
    raft::update_device(secondClusterArray, &arr2[0], nElements, stream);
  }

  void SetupZeroArray() {
    lowerLabelRange = 0;
    upperLabelRange = 0;
    truthAdjustedRandIndex = 1.0;
  }

  AdjustedRandIndexParam params;
  T lowerLabelRange, upperLabelRange;
  T *firstClusterArray = nullptr;
  T *secondClusterArray = nullptr;
  int nElements = 0;
  double truthAdjustedRandIndex = 0;
  double computedAdjustedRandIndex = 0;
  cudaStream_t stream;
};

const std::vector<AdjustedRandIndexParam> inputs = {
  {199, 1, 10, false, 0.000001, false},  {200, 15, 100, false, 0.000001, false},
  {100, 1, 20, false, 0.000001, false},  {10, 1, 10, false, 0.000001, false},
  {198, 1, 100, false, 0.000001, false}, {300, 3, 99, false, 0.000001, false},
  {199, 1, 10, true, 0.000001, false},   {200, 15, 100, true, 0.000001, false},
  {100, 1, 20, true, 0.000001, false},   {10, 1, 10, true, 0.000001, false},
  {198, 1, 100, true, 0.000001, false},  {300, 3, 99, true, 0.000001, false},

  {199, 0, 0, false, 0.000001, true},    {200, 0, 0, false, 0.000001, true},
  {100, 0, 0, false, 0.000001, true},    {10, 0, 0, false, 0.000001, true},
  {198, 0, 0, false, 0.000001, true},    {300, 0, 0, false, 0.000001, true},
  {199, 0, 0, true, 0.000001, true},     {200, 0, 0, true, 0.000001, true},
  {100, 0, 0, true, 0.000001, true},     {10, 0, 0, true, 0.000001, true},
  {198, 0, 0, true, 0.000001, true},     {300, 0, 0, true, 0.000001, true},
};

const std::vector<AdjustedRandIndexParam> large_inputs = {
  {2000000, 1, 1000, false, 0.000001, false},
  {2000000, 1, 1000, true, 0.000001, false},

  {2000000, 0, 0, false, 0.000001, true},
  {2000000, 0, 0, true, 0.000001, true},
};

typedef AdjustedRandIndexTest<int, int> ARI_ii;
TEST_P(ARI_ii, Result) {
  ASSERT_NEAR(computedAdjustedRandIndex, truthAdjustedRandIndex,
              params.tolerance);
}
INSTANTIATE_TEST_CASE_P(AdjustedRandIndex, ARI_ii, ::testing::ValuesIn(inputs));

typedef AdjustedRandIndexTest<int, unsigned long long> ARI_il;
TEST_P(ARI_il, Result) {
  ASSERT_NEAR(computedAdjustedRandIndex, truthAdjustedRandIndex,
              params.tolerance);
}
INSTANTIATE_TEST_CASE_P(AdjustedRandIndex, ARI_il, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AdjustedRandIndexLarge, ARI_il,
                        ::testing::ValuesIn(large_inputs));

}  //end namespace Metrics
}  //end namespace MLCommon
