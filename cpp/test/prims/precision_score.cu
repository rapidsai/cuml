/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <algorithm>
#include <cuml/common/cuml_allocator.hpp>
#include <iostream>
#include <metrics/precision_score.cuh>
#include <random>
#include "test_utils.h"

namespace MLCommon {
namespace Metrics {

struct precisionScoreParam {
  int nElements;
  int lowerLabelRange;
  int upperLabelRange;
  bool sameArrays;
  double tolerance;
};


//test fixture class
template <typename T>
class precisionScoreTest : public ::testing::TestWithParam<precisionScoreParam> {
protected:
  //the constructor
  void SetUp() override {
    //getting the parameters
    params = ::testing::TestWithParam<precisionScoreParam>::GetParam();

    nElements = params.nElements;
    lowerLabelRange = params.lowerLabelRange;
    upperLabelRange = params.upperLabelRange;

    //generating random value test input
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

    int tp = 0;
    int fp = 0;
    //calculating the true precision score
    for (int i = 0; i <nElements; ++i) {
      tp += arr1[i] == 1 && arr2[i] == 1;
      fp += arr1[i] == 0 && arr2[i] == 1;
    }

    truthPrecisionScore = 0;
    if (tp + fp > 0) {
      truthPrecisionScore = tp / (tp + fp);
    }

    CUDA_CHECK(cudaStreamCreate(&stream));
    raft::allocate(y, nElements, true);
    raft::allocate(y_hat, nElements, true);

    raft::update_device(y, &arr1[0], (int)nElements, stream);
    raft::update_device(y_hat, &arr2[0], (int)nElements, stream);
    std::shared_ptr<MLCommon::deviceAllocator> allocator(
      new raft::mr::device::default_allocator);

    //calling the precision_score CUDA implementation
    computedPrecisionScore = MLCommon::Metrics::precision_score(
      y, y_hat, nElements, allocator, stream);

  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(y_hat));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  //declaring the data values
  precisionScoreParam params;
  T lowerLabelRange, upperLabelRange;
  T *y = nullptr;
  T *y_hat = nullptr;
  int nElements = 0;
  double truthPrecisionScore = 0;
  double computedPrecisionScore = 0;
  cudaStream_t stream;
};

//setting test parameter values
const std::vector<precisionScoreParam> inputs = {
	{28234, 0, 1, false, 0.000001},  {1423, 0, 1, false, 0.000001},
  {0, 0, 1, false, 0.000001},  {3, 0, 1, false, 0.000001},
  {68377, 0, 1, false, 0.000001}, {832, 0, 1, false, 0.000001},
  {550, 0, 1, true, 0.000001},   {304, 0, 1, true, 0.000001},
  {59273, 0, 1, true, 0.000001},   {543, 0, 1, true, 0.000001},
  {31, 0, 1, true, 0.000001},   {7, 0, 1, true, 0.000001},
  {0, 0, 1, true, 0.000001},  {43, 0, 1, true, 0.000001}};

//writing the test suite
typedef precisionScoreTest<int> precisionScoreTestClass;
TEST_P(precisionScoreTestClass, Result) {
  ASSERT_NEAR(computedPrecisionScore, truthPrecisionScore, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(precisionScore, precisionScoreTestClass,
                        ::testing::ValuesIn(inputs));

} //end namespace Metrics
} //end namespace MLCommon