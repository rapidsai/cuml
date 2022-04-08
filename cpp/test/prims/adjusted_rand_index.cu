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

#include "test_utils.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>
#include <metrics/adjusted_rand_index.cuh>
#include <metrics/contingencyMatrix.cuh>
#include <raft/cudart_utils.h>
#include <random>

namespace MLCommon {
namespace Metrics {

struct adjustedRandIndexParam {
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
class adjustedRandIndexTest : public ::testing::TestWithParam<adjustedRandIndexParam> {
 protected:
  adjustedRandIndexTest() : firstClusterArray(0, stream), secondClusterArray(0, stream) {}

  void SetUp() override
  {
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    params    = ::testing::TestWithParam<adjustedRandIndexParam>::GetParam();
    nElements = params.nElements;

    firstClusterArray.resize(nElements, stream);
    secondClusterArray.resize(nElements, stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(firstClusterArray.data(), 0, firstClusterArray.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(secondClusterArray.data(), 0, secondClusterArray.size() * sizeof(T), stream));

    if (!params.testZeroArray) {
      SetUpDifferentArrays();
    } else {
      SetupZeroArray();
    }
    // allocating and initializing memory to the GPU
    computed_adjusted_rand_index = compute_adjusted_rand_index<T, MathT>(
      firstClusterArray.data(), secondClusterArray.data(), nElements, stream);
  }

  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

  void SetUpDifferentArrays()
  {
    lowerLabelRange = params.lowerLabelRange;
    upperLabelRange = params.upperLabelRange;
    std::vector<int> arr1(nElements, 0);
    std::vector<int> arr2(nElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(lowerLabelRange, upperLabelRange);
    std::generate(arr1.begin(), arr1.end(), [&]() { return intGenerator(dre); });
    if (params.sameArrays) {
      arr2 = arr1;
    } else {
      std::generate(arr2.begin(), arr2.end(), [&]() { return intGenerator(dre); });
    }
    // calculating golden output
    int numUniqueClasses = upperLabelRange - lowerLabelRange + 1;
    size_t sizeOfMat     = numUniqueClasses * numUniqueClasses * sizeof(int);
    int* hGoldenOutput   = (int*)malloc(sizeOfMat);
    memset(hGoldenOutput, 0, sizeOfMat);
    for (int i = 0; i < nElements; i++) {
      int row    = arr1[i] - lowerLabelRange;
      int column = arr2[i] - lowerLabelRange;
      hGoldenOutput[row * numUniqueClasses + column] += 1;
    }
    int sumOfNijCTwo = 0;
    int* a           = (int*)malloc(numUniqueClasses * sizeof(int));
    int* b           = (int*)malloc(numUniqueClasses * sizeof(int));
    memset(a, 0, numUniqueClasses * sizeof(int));
    memset(b, 0, numUniqueClasses * sizeof(int));
    int sumOfAiCTwo = 0;
    int sumOfBiCTwo = 0;
    // calculating the sum of number of pairwise points in each index
    // and also the reducing contingency matrix along row and column
    for (int i = 0; i < numUniqueClasses; ++i) {
      for (int j = 0; j < numUniqueClasses; ++j) {
        int Nij = hGoldenOutput[i * numUniqueClasses + j];
        sumOfNijCTwo += ((Nij) * (Nij - 1)) / 2;
        a[i] += hGoldenOutput[i * numUniqueClasses + j];
        b[i] += hGoldenOutput[j * numUniqueClasses + i];
      }
    }
    // claculating the sum of number pairwise points in ever column sum
    // claculating the sum of number pairwise points in ever row sum
    for (int i = 0; i < numUniqueClasses; ++i) {
      sumOfAiCTwo += ((a[i]) * (a[i] - 1)) / 2;
      sumOfBiCTwo += ((b[i]) * (b[i] - 1)) / 2;
    }
    // calculating the ARI
    double nCTwo         = double(nElements) * double(nElements - 1) / 2.0;
    double expectedIndex = (double(sumOfBiCTwo) * double(sumOfAiCTwo)) / double(nCTwo);
    double maxIndex      = (double(sumOfAiCTwo) + double(sumOfBiCTwo)) / 2.0;
    double index         = (double)sumOfNijCTwo;
    if (maxIndex - expectedIndex)
      truth_adjusted_rand_index = (index - expectedIndex) / (maxIndex - expectedIndex);
    else
      truth_adjusted_rand_index = 0;
    raft::update_device(firstClusterArray.data(), &arr1[0], nElements, stream);
    raft::update_device(secondClusterArray.data(), &arr2[0], nElements, stream);
  }

  void SetupZeroArray()
  {
    lowerLabelRange           = 0;
    upperLabelRange           = 0;
    truth_adjusted_rand_index = 1.0;
  }

  adjustedRandIndexParam params;
  T lowerLabelRange, upperLabelRange;
  rmm::device_uvector<T> firstClusterArray;
  rmm::device_uvector<T> secondClusterArray;
  int nElements                       = 0;
  double truth_adjusted_rand_index    = 0;
  double computed_adjusted_rand_index = 0;
  cudaStream_t stream                 = 0;
};

const std::vector<adjustedRandIndexParam> inputs = {
  {199, 1, 10, false, 0.000001, false},
  {200, 15, 100, false, 0.000001, false},
  {100, 1, 20, false, 0.000001, false},
  {10, 1, 10, false, 0.000001, false},
  {198, 1, 100, false, 0.000001, false},
  {300, 3, 99, false, 0.000001, false},
  {199, 1, 10, true, 0.000001, false},
  {200, 15, 100, true, 0.000001, false},
  {100, 1, 20, true, 0.000001, false},
  // FIXME: disabled temporarily due to flaky test
  // {10, 1, 10, true, 0.000001, false},
  {198, 1, 100, true, 0.000001, false},
  {300, 3, 99, true, 0.000001, false},

  {199, 0, 0, false, 0.000001, true},
  {200, 0, 0, false, 0.000001, true},
  {100, 0, 0, false, 0.000001, true},
  {10, 0, 0, false, 0.000001, true},
  {198, 0, 0, false, 0.000001, true},
  {300, 0, 0, false, 0.000001, true},
  {199, 0, 0, true, 0.000001, true},
  {200, 0, 0, true, 0.000001, true},
  {100, 0, 0, true, 0.000001, true},
  {10, 0, 0, true, 0.000001, true},
  {198, 0, 0, true, 0.000001, true},
  {300, 0, 0, true, 0.000001, true},
};

const std::vector<adjustedRandIndexParam> large_inputs = {
  {2000000, 1, 1000, false, 0.000001, false},
  {2000000, 1, 1000, true, 0.000001, false},

  {2000000, 0, 0, false, 0.000001, true},
  {2000000, 0, 0, true, 0.000001, true},
};

typedef adjustedRandIndexTest<int, int> ARI_ii;
TEST_P(ARI_ii, Result)
{
  ASSERT_NEAR(computed_adjusted_rand_index, truth_adjusted_rand_index, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(adjusted_rand_index, ARI_ii, ::testing::ValuesIn(inputs));

typedef adjustedRandIndexTest<int, unsigned long long> ARI_il;
TEST_P(ARI_il, Result)
{
  ASSERT_NEAR(computed_adjusted_rand_index, truth_adjusted_rand_index, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(adjusted_rand_index, ARI_il, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(adjusted_rand_index_large, ARI_il, ::testing::ValuesIn(large_inputs));

}  // end namespace Metrics
}  // end namespace MLCommon
