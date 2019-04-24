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

    size_t workspaceSz = MLCommon::Metrics::getWorkspaceSize(numElements, dY,
                                                      stream, lowerLabelRange, upperLabelRange);
    
    if (workspaceSz != 0)
      MLCommon::allocate(pWorkspace, workspaceSz);

    MLCommon::updateDeviceAsync(dYHat, &y_hat[0], numElements, stream);
    MLCommon::updateDeviceAsync(dY, &y[0], numElements, stream);
    MLCommon::updateDeviceAsync(dGoldenOutput, hGoldenOutput, 
                                  numUniqueClasses*numUniqueClasses, stream);

    if (params.calcCardinality)
      MLCommon::Metrics::contingencyMatrix(dY, dYHat, numElements, dComputedOutput,
                                            stream, (void*)pWorkspace, workspaceSz);
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
  T* dY=NULL;
  T* dYHat=NULL;
  int *dComputedOutput = NULL;
  int *dGoldenOutput = NULL;
  int *hGoldenOutput = NULL;
  char *pWorkspace = NULL;
  cudaStream_t stream;
};

const std::vector<contingencyMatrixParam> inputs = {
  {10000, 1, 10, true, 0.000001},
  {100000, 1, 100, false, 0.000001},
  {1000000, 1, 1200, true, 0.000001},
  {1000000, 1, 10000, false, 0.000001}
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
 
 