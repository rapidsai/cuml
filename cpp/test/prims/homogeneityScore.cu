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
#include <iostream>
#include <random>
#include "common/cuml_allocator.hpp"
#include "metrics/homogeneityScore.h"
#include "test_utils.h"

namespace MLCommon {
namespace Metrics {

//parameter structure definition
struct homogeneityParam {
  int nElements;
  int lowerLabelRange;
  int upperLabelRange;
  bool sameArrays;
  double tolerance;
};

//test fixture class
template <typename T>
class homogeneityTest : public ::testing::TestWithParam<homogeneityParam> {
 protected:
  //the constructor
  void SetUp() override {
    //getting the parameters
    params = ::testing::TestWithParam<homogeneityParam>::GetParam();

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

    //allocating and initializing memory to the GPU

    CUDA_CHECK(cudaStreamCreate(&stream));
    MLCommon::allocate(truthClusterArray, nElements, true);
    MLCommon::allocate(predClusterArray, nElements, true);

    MLCommon::updateDevice(truthClusterArray, &arr1[0], (int)nElements, stream);
    MLCommon::updateDevice(predClusterArray, &arr2[0], (int)nElements, stream);
    std::shared_ptr<MLCommon::deviceAllocator> allocator(
      new defaultDeviceAllocator);

    //calculating the golden output
    double truthMI, truthEntropy;

    truthMI = MLCommon::Metrics::mutualInfoScore(
      truthClusterArray, predClusterArray, nElements, lowerLabelRange,
      upperLabelRange, allocator, stream);
    truthEntropy =
      MLCommon::Metrics::entropy(truthClusterArray, nElements, lowerLabelRange,
                                 upperLabelRange, allocator, stream);

    if (truthEntropy) {
      truthHomogeneity = truthMI / truthEntropy;
    } else
      truthHomogeneity = 1.0;

    if (nElements == 0) truthHomogeneity = 1.0;

    //calling the homogeneity CUDA implementation
    computedHomogeneity = MLCommon::Metrics::homogeneityScore(
      truthClusterArray, predClusterArray, nElements, lowerLabelRange,
      upperLabelRange, allocator, stream);

    printf("computed: %f truth:%f\n entropy:%f mi:%f", computedHomogeneity,
           truthHomogeneity, truthEntropy, truthMI);
  }

  //the destructor
  void TearDown() override {
    CUDA_CHECK(cudaFree(truthClusterArray));
    CUDA_CHECK(cudaFree(predClusterArray));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  //declaring the data values
  homogeneityParam params;
  T lowerLabelRange, upperLabelRange;
  T* truthClusterArray = nullptr;
  T* predClusterArray = nullptr;
  int nElements = 0;
  double truthHomogeneity = 0;
  double computedHomogeneity = 0;
  cudaStream_t stream;
};

//setting test parameter values
const std::vector<homogeneityParam> inputs = {
  {199, 1, 10, false, 0.000001},  {200, 15, 100, false, 0.000001},
  {100, 1, 20, false, 0.000001},  {10, 1, 10, false, 0.000001},
  {198, 1, 100, false, 0.000001}, {300, 3, 99, false, 0.000001},
  {199, 1, 10, true, 0.000001},   {200, 15, 100, true, 0.000001},
  {100, 1, 20, true, 0.000001},   {10, 1, 10, true, 0.000001},
  {198, 1, 100, true, 0.000001},  {300, 3, 99, true, 0.000001}};

//writing the test suite
typedef homogeneityTest<int> homogeneityTestClass;
TEST_P(homogeneityTestClass, Result) {
  ASSERT_NEAR(computedHomogeneity, truthHomogeneity, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(homogeneity, homogeneityTestClass,
                        ::testing::ValuesIn(inputs));

}  //end namespace Metrics
}  //end namespace MLCommon
