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
#include <metrics/v_measure.cuh>
#include <raft/cudart_utils.h>
#include <random>

namespace MLCommon {
namespace Metrics {

// parameter structure definition
struct vMeasureParam {
  int nElements;
  int lowerLabelRange;
  int upperLabelRange;
  double beta;
  bool sameArrays;
  double tolerance;
};

// test fixture class
template <typename T>
class vMeasureTest : public ::testing::TestWithParam<vMeasureParam> {
 protected:
  // the constructor
  void SetUp() override
  {
    // getting the parameters
    params = ::testing::TestWithParam<vMeasureParam>::GetParam();

    nElements       = params.nElements;
    lowerLabelRange = params.lowerLabelRange;
    upperLabelRange = params.upperLabelRange;

    // generating random value test input
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

    // allocating and initializing memory to the GPU

    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    rmm::device_uvector<T> truthClusterArray(nElements, stream);
    rmm::device_uvector<T> predClusterArray(nElements, stream);
    raft::update_device(truthClusterArray.data(), &arr1[0], (int)nElements, stream);
    raft::update_device(predClusterArray.data(), &arr2[0], (int)nElements, stream);

    // calculating the golden output
    double truthHomogeity, truthCompleteness;

    truthHomogeity    = MLCommon::Metrics::homogeneity_score(truthClusterArray.data(),
                                                          predClusterArray.data(),
                                                          nElements,
                                                          lowerLabelRange,
                                                          upperLabelRange,
                                                          stream);
    truthCompleteness = MLCommon::Metrics::homogeneity_score(predClusterArray.data(),
                                                             truthClusterArray.data(),
                                                             nElements,
                                                             lowerLabelRange,
                                                             upperLabelRange,
                                                             stream);

    if (truthCompleteness + truthHomogeity == 0.0)
      truthVMeasure = 0.0;
    else
      truthVMeasure = ((1 + params.beta) * truthHomogeity * truthCompleteness /
                       (params.beta * truthHomogeity + truthCompleteness));
    // calling the v_measure CUDA implementation
    computedVMeasure = MLCommon::Metrics::v_measure(truthClusterArray.data(),
                                                    predClusterArray.data(),
                                                    nElements,
                                                    lowerLabelRange,
                                                    upperLabelRange,
                                                    stream,
                                                    params.beta);
  }

  // the destructor
  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

  // declaring the data values
  vMeasureParam params;
  T lowerLabelRange, upperLabelRange;
  int nElements           = 0;
  double truthVMeasure    = 0;
  double computedVMeasure = 0;
  cudaStream_t stream     = 0;
};

// setting test parameter values
const std::vector<vMeasureParam> inputs = {{199, 1, 10, 1.0, false, 0.000001},
                                           {200, 15, 100, 1.0, false, 0.000001},
                                           {100, 1, 20, 1.0, false, 0.000001},
                                           {10, 1, 10, 1.0, false, 0.000001},
                                           {198, 1, 100, 1.0, false, 0.000001},
                                           {300, 3, 99, 1.0, false, 0.000001},
                                           {199, 1, 10, 1.0, true, 0.000001},
                                           {200, 15, 100, 1.0, true, 0.000001},
                                           {100, 1, 20, 1.0, true, 0.000001},
                                           {10, 1, 10, 1.0, true, 0.000001},
                                           {198, 1, 100, 1.0, true, 0.000001},
                                           {300, 3, 99, 1.0, true, 0.000001}};

// writing the test suite
typedef vMeasureTest<int> vMeasureTestClass;
TEST_P(vMeasureTestClass, Result)
{
  ASSERT_NEAR(computedVMeasure, truthVMeasure, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(vMeasure, vMeasureTestClass, ::testing::ValuesIn(inputs));

}  // end namespace Metrics
}  // end namespace MLCommon
