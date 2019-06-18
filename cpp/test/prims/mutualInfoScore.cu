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
#include "metrics/mutualInfoScore.h"
#include "common/cuml_allocator.hpp"
#include "metrics/contingencyMatrix.h"



namespace MLCommon{
namespace Metrics{

//parameter structure definition
struct mutualInfoParam{

  int nElements;
  int lowerLabelRange;
  int upperLabelRange;
  bool sameArrays;
  double tolerance;

};

//test fixture class
template <typename T>
class mutualInfoTest : public ::testing::TestWithParam<mutualInfoParam>{
  protected:
  //the constructor
  void SetUp() override {

    //getting the parameters
    params = ::testing::TestWithParam<mutualInfoParam>::GetParam();

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

    int *a = (int *)malloc(numUniqueClasses*sizeof(int));
    int *b = (int *)malloc(numUniqueClasses*sizeof(int));
    memset(a, 0, numUniqueClasses*sizeof(int));
    memset(b, 0, numUniqueClasses*sizeof(int));

    
    //and also the reducing contingency matrix along row and column
    for(i=0;i<numUniqueClasses;++i){
        for(j=0;j<numUniqueClasses;++j){
            a[i]+=hGoldenOutput[i*numUniqueClasses + j];
            b[i]+=hGoldenOutput[j*numUniqueClasses + i];
        }
    }


//calculating the truth mutual information
    for(int i =0; i<numUniqueClasses; ++i){
        for(int j = 0; j<numUniqueClasses; ++j){

            if(a[i]*b[j]!=0 && hGoldenOutput[i*numUniqueClasses + j]!=0){

            truthmutualInfo+= (double)(hGoldenOutput[i*numUniqueClasses + j])*double(log((double)(hGoldenOutput[i*numUniqueClasses + j]))-log((double)(a[i]*b[j])));

            }

        }
    }

    

    //allocating and initializing memory to the GPU
    CUDA_CHECK(cudaStreamCreate(&stream));
    MLCommon::allocate(firstClusterArray,nElements,true);
    MLCommon::allocate(secondClusterArray,nElements,true);

    MLCommon::updateDevice(firstClusterArray,&arr1[0],(int)nElements,stream);
    MLCommon::updateDevice(secondClusterArray,&arr2[0],(int)nElements,stream);
    std::shared_ptr<MLCommon::deviceAllocator> allocator(new defaultDeviceAllocator);


    //calling the mutualInfo CUDA implementation
    computedmutualInfo = MLCommon::Metrics::mutualInfoScore(firstClusterArray,secondClusterArray,nElements, lowerLabelRange, upperLabelRange, allocator,stream);

    }

    //the destructor
    void TearDown() override
    {
        
        CUDA_CHECK(cudaFree(firstClusterArray));
        CUDA_CHECK(cudaFree(secondClusterArray));
        CUDA_CHECK(cudaStreamDestroy(stream));


    }

    //declaring the data values
    mutualInfoParam params;
    T lowerLabelRange,upperLabelRange;
    T* firstClusterArray=nullptr;
    T* secondClusterArray = nullptr;
    int nElements=0;
    double truthmutualInfo=0;
    double computedmutualInfo = 0;
    cudaStream_t stream;

    };

//setting test parameter values
const std::vector<mutualInfoParam> inputs = {
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
typedef mutualInfoTest<int> mutualInfoTestClass;
TEST_P(mutualInfoTestClass, Result){
    ASSERT_NEAR(computedmutualInfo, truthmutualInfo, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(mutualInfo, mutualInfoTestClass,::testing::ValuesIn(inputs));


}//end namespace Metrics
}//end namespace MLCommon
