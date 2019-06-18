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
#include "metrics/silhouetteScore.h"
#include "test_utils.h"

namespace MLCommon {
namespace Metrics {

//parameter structure definition
struct silhouetteScoreParam {
  int nRows;
  int nCols;
  int nLabels;
  int metric;
  double tolerance;
};

//test fixture class
template <typename LabelT, typename DataT>
class silhouetteScoreTest
  : public ::testing::TestWithParam<silhouetteScoreParam> {
 protected:
  //the constructor
  void SetUp() override {
    //getting the parameters
    params = ::testing::TestWithParam<silhouetteScoreParam>::GetParam();

    nRows = params.nRows;
    nCols = params.nCols;
    nLabels = params.nLabels;
    int nElements = nRows * nCols;

    //generating random value test input
    std::vector<double> h_X = {0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0};
    h_X.resize(nRows * nCols);
    std::vector<int> h_labels = {0, 1, 0, 1};
    h_labels.resize(nRows);
    /*std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(lowerLabelRange, upperLabelRange);

    std::generate()
    std::generate(h_labels.begin(), h_X.end(), [&](){return intGenerator(dre); });*/

    //generating the golden output

    //calculating the distance matrix

    truthSilhouetteScore = 3.5 / 4.5;

    //allocating and initializing memory to the GPU
    CUDA_CHECK(cudaStreamCreate(&stream));
    MLCommon::allocate(d_X, nElements, true);
    MLCommon::allocate(d_labels, nElements, true);

    MLCommon::updateDevice(d_X, &h_X[0], (int)nElements, stream);
    MLCommon::updateDevice(d_labels, &h_labels[0], (int)nElements, stream);
    std::shared_ptr<MLCommon::deviceAllocator> allocator(
      new defaultDeviceAllocator);

    //calling the silhouetteScore CUDA implementation
    computedSilhouetteScore = MLCommon::Metrics::silhouetteScore(
      d_X, nRows, nCols, d_labels, nLabels, sampleSilScore, allocator, stream,
      params.metric);
  }

  //the destructor
  void TearDown() override {
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  //declaring the data values
  silhouetteScoreParam params;
  int nLabels;
  DataT* d_X = nullptr;
  DataT* sampleSilScore = nullptr;
  LabelT* d_labels = nullptr;
  int nRows;
  int nCols;
  double truthSilhouetteScore = 0;
  double computedSilhouetteScore = 0;
  cudaStream_t stream;
};

//setting test parameter values
const std::vector<silhouetteScoreParam> inputs = {{4, 2, 2, 4, 0.00001}};

//writing the test suite
typedef silhouetteScoreTest<int, double> silhouetteScoreTestClass;
TEST_P(silhouetteScoreTestClass, Result) {
  ASSERT_NEAR(computedSilhouetteScore, truthSilhouetteScore, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(silhouetteScore, silhouetteScoreTestClass,
                        ::testing::ValuesIn(inputs));

}  //end namespace Metrics
}  //end namespace MLCommon
