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
 #include "test_utils.h"
 #include <iostream>
 #include <random>
 #include <algorithm>
 #include "metrics/contingencyMatrix.h"
 
namespace MLCommon {
namespace Metrics {

struct contingencyMatrixParam {
  int nElements;
  int minClass;
  int maxClass;
  bool calcCardinality;
  bool skipLabels;
  float tolerance;
};

template <typename T>
class ContingencyMatrixTestImpl : public ::testing::TestWithParam<contingencyMatrixParam> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<contingencyMatrixParam>::GetParam();

    int numElements = params.nElements;
    int lowerLabelRange = params.minClass;
    int upperLabelRange = params.maxClass;
    
    std::vector<int> y(numElements, 0);
    std::vector<int> y_hat(numElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(lowerLabelRange, upperLabelRange);

    std::generate(y.begin(), y.end(), [&](){return intGenerator(dre); });
    std::generate(y_hat.begin(), y_hat.end(), [&](){return intGenerator(dre); });

    if (params.skipLabels) {
      // remove two label value from input arrays
      int y1 = (upperLabelRange - lowerLabelRange) / 2;
      int y2 = y1  + (upperLabelRange - lowerLabelRange) / 4;

      // replacement values
      int y1_R = y1 + 1;
      int y2_R = y2 + 1;

      std::replace(y.begin(), y.end(), y1, y1_R);
      std::replace(y.begin(), y.end(), y2, y2_R);
      std::replace(y_hat.begin(), y_hat.end(), y1, y1_R);
      std::replace(y_hat.begin(), y_hat.end(), y2, y2_R);
    }

    numUniqueClasses = upperLabelRange - lowerLabelRange + 1;

    // generate golden output on CPU
    size_t sizeOfMat = numUniqueClasses*numUniqueClasses * sizeof(int);
    int *hGoldenOutput = (int *)malloc(sizeOfMat);
    memset(hGoldenOutput, 0, sizeOfMat);

    for (int i = 0; i < numElements; i++) {
      int row = y[i] - lowerLabelRange;
      int column = y_hat[i] - lowerLabelRange;

      hGoldenOutput[row * numUniqueClasses + column] += 1;
    }
    
    CUDA_CHECK(cudaStreamCreate(&stream));
    MLCommon::allocate(dY, numElements);
    MLCommon::allocate(dYHat, numElements);
    MLCommon::allocate(dComputedOutput, numUniqueClasses*numUniqueClasses);
    MLCommon::allocate(dGoldenOutput, numUniqueClasses*numUniqueClasses);

    size_t workspaceSz = MLCommon::Metrics::getCMatrixWorkspaceSize(numElements, dY,
                                                      stream, lowerLabelRange, upperLabelRange);
    
    if (workspaceSz != 0)
      MLCommon::allocate(pWorkspace, workspaceSz);

    MLCommon::updateDevice(dYHat, &y_hat[0], numElements, stream);
    MLCommon::updateDevice(dY, &y[0], numElements, stream);
    MLCommon::updateDevice(dGoldenOutput, hGoldenOutput, 
                                  numUniqueClasses*numUniqueClasses, stream);

    if (params.calcCardinality) {
      T minLabel, maxLabel;
      MLCommon::Metrics::getInputClassCardinality(dY, numElements, stream, minLabel, maxLabel);
      // allocate dComputedOutput using minLabel, maxLabel count - already done above
      MLCommon::Metrics::contingencyMatrix(dY, dYHat, numElements, dComputedOutput,
                                            stream, (void*)pWorkspace, workspaceSz,
                                            minLabel, maxLabel);
    }
    else
      MLCommon::Metrics::contingencyMatrix(dY, dYHat, numElements, dComputedOutput,
                                            stream, (void*)pWorkspace, workspaceSz, 
                                            lowerLabelRange, upperLabelRange);
  }

  void TearDown() override {
    free(hGoldenOutput);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dYHat));
    CUDA_CHECK(cudaFree(dComputedOutput));
    CUDA_CHECK(cudaFree(dGoldenOutput));
    if (pWorkspace)
      CUDA_CHECK(cudaFree(pWorkspace));
  }

  contingencyMatrixParam params;
  int numUniqueClasses = -1;
  T* dY=nullptr;
  T* dYHat=nullptr;
  int *dComputedOutput = nullptr;
  int *dGoldenOutput = nullptr;
  int *hGoldenOutput = nullptr;
  char *pWorkspace = nullptr;
  cudaStream_t stream;
};

const std::vector<contingencyMatrixParam> inputs = {
  {10000, 1, 10, true, false, 0.000001},
  {100000, 1, 100, false, false, 0.000001},
  {1000000, 1, 1200, true, false, 0.000001},
  {1000000, 1, 10000, false, false, 0.000001},
  {100000, 1, 100, false, true, 0.000001}
};

typedef ContingencyMatrixTestImpl<int> ContingencyMatrixTestImplS; 
TEST_P(ContingencyMatrixTestImplS, Result) {
  ASSERT_TRUE(devArrMatch(dComputedOutput, dGoldenOutput, numUniqueClasses * numUniqueClasses,
                  CompareApprox<float>(params.tolerance)));
}
  
INSTANTIATE_TEST_CASE_P(ContingencyMatrix, ContingencyMatrixTestImplS,
                ::testing::ValuesIn(inputs));
}
}
 
 