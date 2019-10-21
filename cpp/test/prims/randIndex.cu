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
#include <gtest/gtest.h>
#include <algorithm>
#include <cuml/common/cuml_allocator.hpp>
#include <iostream>
#include <random>
#include "metrics/randIndex.h"
#include "test_utils.h"

namespace MLCommon {
namespace Metrics {

//parameter structure definition
struct randIndexParam {
  uint64_t nElements;
  int lowerLabelRange;
  int upperLabelRange;
  double tolerance;
};

//test fixture class
template <typename T>
class randIndexTest : public ::testing::TestWithParam<randIndexParam> {
 protected:
  //the constructor
  void SetUp() override {
    //getting the parameters
    params = ::testing::TestWithParam<randIndexParam>::GetParam();

    size = params.nElements;
    lowerLabelRange = params.lowerLabelRange;
    upperLabelRange = params.upperLabelRange;

    //generating random value test input
    std::vector<int> arr1(size, 0);
    std::vector<int> arr2(size, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(lowerLabelRange,
                                                    upperLabelRange);

    std::generate(arr1.begin(), arr1.end(),
                  [&]() { return intGenerator(dre); });
    std::generate(arr2.begin(), arr2.end(),
                  [&]() { return intGenerator(dre); });

    //generating the golden output
    int64_t a_truth = 0, b_truth = 0, iter = 0, jiter;
    for (; iter < size; ++iter) {
      for (jiter = 0; jiter < iter; ++jiter) {
        if (arr1[iter] == arr1[jiter] && arr2[iter] == arr2[jiter]) {
          ++a_truth;
        } else if (arr1[iter] != arr1[jiter] && arr2[iter] != arr2[jiter]) {
          ++b_truth;
        }
      }
    }
    uint64_t nChooseTwo = (size * (size - 1)) / 2;
    truthRandIndex =
      (double)(((double)(a_truth + b_truth)) / (double)nChooseTwo);

    //allocating and initializing memory to the GPU
    CUDA_CHECK(cudaStreamCreate(&stream));
    MLCommon::allocate(firstClusterArray, size, true);
    MLCommon::allocate(secondClusterArray, size, true);

    MLCommon::updateDevice(firstClusterArray, &arr1[0], (int)size, stream);
    MLCommon::updateDevice(secondClusterArray, &arr2[0], (int)size, stream);
    std::shared_ptr<MLCommon::deviceAllocator> allocator(
      new defaultDeviceAllocator);

    //calling the randIndex CUDA implementation
    computedRandIndex = MLCommon::Metrics::computeRandIndex(
      firstClusterArray, secondClusterArray, size, allocator, stream);
  }

  //the destructor
  void TearDown() override {
    CUDA_CHECK(cudaFree(firstClusterArray));
    CUDA_CHECK(cudaFree(secondClusterArray));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  //declaring the data values
  randIndexParam params;
  int lowerLabelRange = 0, upperLabelRange = 2;
  T* firstClusterArray = nullptr;
  T* secondClusterArray = nullptr;
  uint64_t size = 0;
  double truthRandIndex = 0;
  double computedRandIndex = 0;
  cudaStream_t stream;
};

//setting test parameter values
const std::vector<randIndexParam> inputs = {
  {199, 1, 10, 0.000001},    {200, 1, 100, 0.000001}, {10, 1, 1200, 0.000001},
  {100, 1, 10000, 0.000001}, {198, 1, 100, 0.000001}, {300, 3, 99, 0.000001},
  {2, 0, 0, 0.00001}};

//writing the test suite
typedef randIndexTest<int> randIndexTestClass;
TEST_P(randIndexTestClass, Result) {
  ASSERT_NEAR(computedRandIndex, truthRandIndex, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(randIndex, randIndexTestClass,
                        ::testing::ValuesIn(inputs));

}  //end namespace Metrics
}  //end namespace MLCommon
