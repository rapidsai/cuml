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
#include "metrics/adjustedRandIndex.h"
#include "common/cuml_allocator.hpp"
#include "metrics/contingencyMatrix.h"



namespace MLCommon{
namespace Metrics{

//parameter structure definition
struct AdjustedRandIndexParam{

  int nElements;
  int lowerLabelRange;
  int upperLabelRange;
  bool sameArrays;
  double tolerance;

};

//test fixture class
template <typename T>
class adjustedRandIndexTest : public ::testing::TestWithParam<AdjustedRandIndexParam>{
  protected:
  //the constructor
  void SetUp() override {

    //getting the parameters
    params = ::testing::TestWithParam<AdjustedRandIndexParam>::GetParam();

    nElements = params.nElements;
    lowerLabelRange = params.lowerLabelRange;
    upperLabelRange = params.upperLabelRange;

    //generating random value test input
    std::vector<int> arr1(nElements, 0);
    std::vector<int> arr2(nElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(lowerLabelRange, upperLabelRange);

    std::generate(arr1.begin(), arr1.end(), [&](){return intGenerator(dre); });
    if(params.sameArrays) {
        arr2 = arr1;
    } else {
        std::generate(arr2.begin(), arr2.end(), [&](){return intGenerator(dre); });
    }

    //generating the golden output
    //calculating the contingency matrix
    int numUniqueClasses = upperLabelRange - lowerLabelRange + 1;
    size_t sizeOfMat = numUniqueClasses*numUniqueClasses * sizeof(int);
    int *hGoldenOutput = (int *)malloc(sizeOfMat);
    memset(hGoldenOutput, 0, sizeOfMat);
    int i,j;
    for (i = 0; i < nElements; i++) {
      int row = arr1[i] - lowerLabelRange;
      int column = arr2[i] - lowerLabelRange;

      hGoldenOutput[row * numUniqueClasses + column] += 1;
    }    
    int sumOfNijCTwo = 0;
    int *a = (int *)malloc(numUniqueClasses*sizeof(int));
    int *b = (int *)malloc(numUniqueClasses*sizeof(int));
    memset(a, 0, numUniqueClasses*sizeof(int));
    memset(b, 0, numUniqueClasses*sizeof(int));
    int sumOfAiCTwo = 0;
    int sumOfBiCTwo = 0;

    //calculating the sum of number of pairwise points in each index 
    //and also the reducing contingency matrix along row and column
    for(i=0;i<numUniqueClasses;++i){
        for(j=0;j<numUniqueClasses;++j){
            int Nij = hGoldenOutput[i*numUniqueClasses + j];
            sumOfNijCTwo += ((Nij)*(Nij-1))/2;
            a[i]+=hGoldenOutput[i*numUniqueClasses + j];
            b[i]+=hGoldenOutput[j*numUniqueClasses + i];
        }
    }

    //claculating the sum of number pairwise points in ever column sum
    //claculating the sum of number pairwise points in ever row sum
    for(i=0;i<numUniqueClasses;++i){
        sumOfAiCTwo += ((a[i])*(a[i]-1))/2;
        sumOfBiCTwo += ((b[i])*(b[i]-1))/2;
    }

    //calculating the ARI
    int nCTwo = ((nElements)*(nElements-1))/2;
    double expectedIndex = ((double)(sumOfBiCTwo*sumOfAiCTwo))/((double)(nCTwo));
    double maxIndex = ((double)(sumOfAiCTwo+sumOfBiCTwo))/2.0;
    double index = (double)sumOfNijCTwo;

    if(maxIndex - expectedIndex)
        truthAdjustedRandIndex = (index - expectedIndex)/(maxIndex - expectedIndex);
    else truthAdjustedRandIndex = 0;

    //allocating and initializing memory to the GPU
    CUDA_CHECK(cudaStreamCreate(&stream));
    MLCommon::allocate(firstClusterArray,nElements,true);
    MLCommon::allocate(secondClusterArray,nElements,true);

    MLCommon::updateDevice(firstClusterArray,&arr1[0],(int)nElements,stream);
    MLCommon::updateDevice(secondClusterArray,&arr2[0],(int)nElements,stream);
    std::shared_ptr<MLCommon::deviceAllocator> allocator(new defaultDeviceAllocator);


    //calling the adjustedRandIndex CUDA implementation
    computedAdjustedRandIndex = MLCommon::Metrics::computeAdjustedRandIndex(firstClusterArray,secondClusterArray,nElements, lowerLabelRange, upperLabelRange, allocator,stream);

    }

    //the destructor
    void TearDown() override
    {
        
        CUDA_CHECK(cudaFree(firstClusterArray));
        CUDA_CHECK(cudaFree(secondClusterArray));
        CUDA_CHECK(cudaStreamDestroy(stream));


    }

    //declaring the data values
    AdjustedRandIndexParam params;
    T lowerLabelRange,upperLabelRange;
    T* firstClusterArray=nullptr;
    T* secondClusterArray = nullptr;
    int nElements=0;
    double truthAdjustedRandIndex=0;
    double computedAdjustedRandIndex = 0;
    cudaStream_t stream;

    };

//setting test parameter values
const std::vector<AdjustedRandIndexParam> inputs = {
    {199, 1, 10, false, 0.000001},
    {200, 15, 100, false, 0.000001},
    {100, 1, 20, false, 0.000001},
    {10, 1, 10, false, 0.000001},
   {198, 1, 100, false, 0.000001},
    {300, 3, 99, false, 0.000001},
    {199, 1, 10, true, 0.000001},
    {200, 15, 100, true, 0.000001},
    {100, 1, 20, true, 0.000001},
    {10, 1, 10, true, 0.000001},
   {198, 1, 100, true, 0.000001},
    {300, 3, 99, true, 0.000001}
};


//writing the test suite
typedef adjustedRandIndexTest<int> adjustedRandIndexTestClass;
TEST_P(adjustedRandIndexTestClass, Result){
    ASSERT_NEAR(computedAdjustedRandIndex, truthAdjustedRandIndex, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(adjustedRandIndex, adjustedRandIndexTestClass,::testing::ValuesIn(inputs));


}//end namespace Metrics
}//end namespace MLCommon
