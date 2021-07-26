/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <iostream>
#include <metrics/contingencyMatrix.cuh>
#include <random>
#include "test_utils.h"

namespace MLCommon {
namespace Metrics {

struct ContingencyMatrixParam {
  int nElements;
  int minClass;
  int maxClass;
  bool calcCardinality;
  bool skipLabels;
  float tolerance;
};

template <typename T>
class ContingencyMatrixTest : public ::testing::TestWithParam<ContingencyMatrixParam> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<ContingencyMatrixParam>::GetParam();

    int numElements     = params.nElements;
    int lowerLabelRange = params.minClass;
    int upperLabelRange = params.maxClass;

    std::vector<int> y(numElements, 0);
    std::vector<int> y_hat(numElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(lowerLabelRange, upperLabelRange);

    std::generate(y.begin(), y.end(), [&]() { return intGenerator(dre); });
    std::generate(y_hat.begin(), y_hat.end(), [&]() { return intGenerator(dre); });

    if (params.skipLabels) {
      // remove two label value from input arrays
      int y1 = (upperLabelRange - lowerLabelRange) / 2;
      int y2 = y1 + (upperLabelRange - lowerLabelRange) / 4;

      // replacement values
      int y1_R = y1 + 1;
      int y2_R = y2 + 1;

      std::replace(y.begin(), y.end(), y1, y1_R);
      std::replace(y.begin(), y.end(), y2, y2_R);
      std::replace(y_hat.begin(), y_hat.end(), y1, y1_R);
      std::replace(y_hat.begin(), y_hat.end(), y2, y2_R);
    }

    CUDA_CHECK(cudaStreamCreate(&stream));
    dY =  = std::make_unique<rmm::device_uvector<T>>(nElements, stream);
    dYHat = = std::make_unique<rmm::device_uvector<T>>(nElements, stream);

    raft::update_device(dYHat->data(), &y_hat[0], numElements, stream);
    raft::update_device(dY->data(), &y[0], numElements, stream);

    if (params.calcCardinality) {
      MLCommon::Metrics::getInputClassCardinality(
        dY->data(), numElements, stream, minLabel, maxLabel);
    } else {
      minLabel = lowerLabelRange;
      maxLabel = upperLabelRange;
    }

    numUniqueClasses = maxLabel - minLabel + 1;

    dComputedOutput = =
      std::make_unique<rmm::device_uvector<int>>(numUniqueClasses * numUniqueClasses, stream);
    dGoldenOutput = =
      std::make_unique<rmm::device_uvector<int>>(numUniqueClasses * numUniqueClasses, stream);

    // generate golden output on CPU
    size_t sizeOfMat = numUniqueClasses * numUniqueClasses * sizeof(int);
    std::vector<int> hGoldenOutput(sizeOfMat, 0);

    for (int i = 0; i < numElements; i++) {
      auto row    = y[i] - minLabel;
      auto column = y_hat[i] - minLabel;
      hGoldenOutput[row * numUniqueClasses + column] += 1;
    }

    raft::update_device(
      dGoldenOutput->data(), hGoldenOutput->data(), numUniqueClasses * numUniqueClasses, stream);

    workspaceSz = MLCommon::Metrics::getContingencyMatrixWorkspaceSize(
      numElements, dY, stream, minLabel, maxLabel);
    pWorkspace = std::make_unique<rmm::device_uvector<char>>(workspaceSz, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream)); }

  void RunTest()
  {
    int numElements = params.nElements;
    MLCommon::Metrics::contingencyMatrix(dY->data(),
                                         dYHat->data(),
                                         numElements,
                                         dComputedOutput->data(),
                                         stream,
                                         (void*)pWorkspace->data(),
                                         workspaceSz,
                                         minLabel,
                                         maxLabel);
    ASSERT_TRUE(raft::devArrMatch(dComputedOutput->data(),
                                  dGoldenOutput->data(),
                                  numUniqueClasses * numUniqueClasses,
                                  raft::Compare<T>()));
  }

  ContingencyMatrixParam params;
  int numUniqueClasses = -1;
  T minLabel, maxLabel;
  cudaStream_t stream;
  size_t workspaceSz;
  std::unique_ptr<rmm::device_uvector<char>> pWorkspace;
  std::unique_ptr<rmm::device_uvector<T>> dY, dYHat;
  std::unique_ptr<rmm::device_uvector<int>> dComputedOutput, dGoldenOutput;
};

const std::vector<ContingencyMatrixParam> inputs = {
  {10000, 1, 10, true, false, 0.000001},
  {10000, 1, 5000, true, false, 0.000001},
  {10000, 1, 10000, true, false, 0.000001},
  {10000, 1, 20000, true, false, 0.000001},
  {10000, 1, 10, false, false, 0.000001},
  {10000, 1, 5000, false, false, 0.000001},
  {10000, 1, 10000, false, false, 0.000001},
  {10000, 1, 20000, false, false, 0.000001},
  {100000, 1, 100, false, false, 0.000001},
  {1000000, 1, 1200, true, false, 0.000001},
  {1000000, 1, 10000, false, false, 0.000001},
  {100000, 1, 100, false, true, 0.000001},
};

typedef ContingencyMatrixTest<int> ContingencyMatrixTestS;
TEST_P(ContingencyMatrixTestS, Result) { RunTest(); }
INSTANTIATE_TEST_CASE_P(ContingencyMatrix, ContingencyMatrixTestS, ::testing::ValuesIn(inputs));
}  // namespace Metrics
}  // namespace MLCommon
