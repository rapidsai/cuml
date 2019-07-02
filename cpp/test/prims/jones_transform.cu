/*
 * Copyright (c) 2019, NVIDIA CORPORATION. *
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
#include "test_utils.h"
#include "timeSeries/jones_transform.h"

namespace MLCommon {
namespace TimeSeries {

//parameter structure definition
struct JonesTransParam {
  int batchSize;
  int pValue;
  double tolerance;
};

//test fixture class
template

  <typename DataT>
  class JonesTransTest
  : public ::testing::TestWithParam<JonesTransParam> {
 protected:
  //the constructor
  void SetUp() override {
    //getting the parameters
    params = ::testing::TestWithParam<JonesTransParam>::GetParam();

    nElements = params.batchSize* params.pValue;

    //generating random value test input that is stored in row major
    std::vector<double> arr1(nElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_real_distribution<double> realGenerator(0,1);

    std::generate(arr1.begin(), arr1.end(),
                  [&]() { return realGenerator(dre); });


//>>>>>>>>>>>>>>>>> AR transform golden output generation<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    double *newParams = (double*) malloc(nElements*sizeof(double*));
    double *tmp = (double*) malloc(params.pValue*sizeof(double*));

    //for every model in the batch
    for(int i=0; i<params.batchSize; ++i){

      //storing the partial autocorrelation of each ar coefficient of a given batch in newParams and the same in another temporary copy
      for(int j=0; j<params.pValue; ++j){
        newParams[i*params.pValue + j] = ((1-exp(-1*arr1[i*params.pValue + j]))/(1+exp(-1*arr1[i*params.pValue + j])));
        tmp[j] = newParams[i*params.pValue + j];
      }

      //calculating according to jone's recursive formula: phi(j,k) = phi(j-1,k) - a(j)*phi(j-1,j-k)
      for(int j=1; j<params.pValue; ++j){

        //a is partial autocorrelation for jth coefficient
        DataT a = newParams[i*params.pValue + j];

        /*the recursive implementation of the transformation with:
        - lhs tmp[k] => phi(j,k)
        - rhs tmp[k] => phi(j-1,k)
        - a => a(j)
        - newParam[i*params.pValue + j-k-1] => phi(j-1, j-k)
        */
        for(int k = 0; k<j; ++k){

          tmp[k] -= a*newParams[i*params.pValue + (j-k-1)];
        }

        //copying it back for the next iteration
        for(int iter = 0; iter<j; ++iter){

          newParams[i*params.pValue + iter] = tmp[iter];
        }
      }
    }

    //allocating and initializing device memory
    CUDA_CHECK(cudaStreamCreate(&stream));
    MLCommon::allocate(d_golden_ar_trans, nElements, true);
    MLCommon::allocate(d_computed_ar_trans, nElements, true);
    MLCommon::allocate(d_params, nElements, true);

    MLCommon::updateDevice(d_params, &arr1[0], (size_t)nElements, stream);
    MLCommon::updateDevice(d_golden_ar_trans, newParams, (size_t)nElements,
                           stream);
    std::shared_ptr<MLCommon::deviceAllocator> allocator(
      new defaultDeviceAllocator);

    //calling the ar_trans_param CUDA implementation
    MLCommon::TimeSeries::ar_param_transform(
      d_params, params.batchSize, params.pValue, d_computed_ar_trans, allocator, stream);


    //>>>>>>>>>>>>>>>>> MA transform golden output generation<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    //for every model in the batch
    for(int i=0; i<params.batchSize; ++i){

      //storing the partial autocorrelation of each ma coefficient of a given batch in newParams and the same in another temporary copy
      for(int j=0; j<params.pValue; ++j){
        newParams[i*params.pValue + j] = ((1-exp(-1*arr1[i*params.pValue + j]))/(1+exp(-1*arr1[i*params.pValue + j])));
        tmp[j] = newParams[i*params.pValue + j];
      }

      //calculating according to jone's recursive formula: phi(j,k) = phi(j-1,k) - a(j)*phi(j-1,j-k)
      for(int j=1; j<params.pValue; ++j){

        //a is partial autocorrelation for jth coefficient
        DataT a = newParams[i*params.pValue + j];

        /*the recursive implementation of the transformation with:
        - lhs tmp[k] => phi(j,k)
        - rhs tmp[k] => phi(j-1,k)
        - a => a(j)
        - newParam[i*params.pValue + j-k-1] => phi(j-1, j-k)
        */
        for(int k = 0; k<j; ++k){

          tmp[k] += a*newParams[i*params.pValue + (j-k-1)];
        }

        //copying it back for the next iteration
        for(int iter = 0; iter<j; ++iter){

          newParams[i*params.pValue + iter] = tmp[iter];
        }
      }
    }

    //allocating and initializing device memory
    MLCommon::allocate(d_golden_ma_trans, nElements, true);
    MLCommon::allocate(d_computed_ma_trans, nElements, true);

    MLCommon::updateDevice(d_golden_ma_trans, newParams, (size_t)nElements,
                           stream);

    //calling the ma_param_transform CUDA implementation
    MLCommon::TimeSeries::ma_param_transform(
      d_params, params.batchSize, params.pValue, d_computed_ma_trans, allocator, stream);


    //>>>>>>>>>>>>>>>>> AR inverse transform <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    //allocating and initializing device memory
    MLCommon::allocate(d_computed_ar_invtrans, nElements, true);

    //calling the ar_param_inverse_transform CUDA implementation
    MLCommon::TimeSeries::ar_param_inverse_transform(
      d_computed_ar_trans, params.batchSize, params.pValue, d_computed_ar_invtrans, allocator, stream);


    //>>>>>>>>>>>>>>>>> MA inverse transform <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    MLCommon::allocate(d_computed_ma_invtrans, nElements, true);

    //calling the ma_param_inverse_transform CUDA implementation
    MLCommon::TimeSeries::ma_param_inverse_transform(
      d_computed_ma_trans, params.batchSize, params.pValue, d_computed_ma_invtrans, allocator, stream);


  }


  //the destructor
  void TearDown() override {
    CUDA_CHECK(cudaFree(d_computed_ar_trans));
    CUDA_CHECK(cudaFree(d_computed_ma_trans));
    CUDA_CHECK(cudaFree(d_computed_ar_invtrans));
    CUDA_CHECK(cudaFree(d_computed_ma_invtrans));
    CUDA_CHECK(cudaFree(d_golden_ar_trans));
    CUDA_CHECK(cudaFree(d_golden_ma_trans));
    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  //declaring the data values
  JonesTransParam params;
  DataT *d_golden_ar_trans = nullptr;
  DataT *d_golden_ma_trans = nullptr;
  DataT *d_computed_ar_trans = nullptr;
  DataT *d_computed_ma_trans = nullptr;
  DataT *d_computed_ar_invtrans = nullptr;
  DataT *d_computed_ma_invtrans = nullptr;
  DataT *d_params = nullptr;
  cudaStream_t stream;
  int nElements = -1;
};

//setting test parameter values
const std::vector<JonesTransParam> inputs = {
  {500,1,0.001},
  {5000,2,0.001},
  {4,1,0.001},
  {3,3,0.001},
  {5,1,0.001},
  {5,4,0.001},
  {5000,4,0.0001}
};

//writing the test suite
typedef JonesTransTest<double> JonesTransTestClass;
TEST_P(JonesTransTestClass, Result) {
  ASSERT_TRUE(devArrMatch(d_computed_ar_trans, d_golden_ar_trans, nElements, CompareApprox<double>(params.tolerance)));
  ASSERT_TRUE(devArrMatch(d_computed_ma_trans, d_golden_ma_trans, nElements, CompareApprox<double>(params.tolerance)));
  /*
  Test verifying the inversion property:
  initially generated random coefficients -> ar_param_transform() / ma_param_transform() -> 
  transformed coefficients -> ar_param_inverse_transform()/ma_param_inverse_transform() -> 
  initially generated random coefficients
  */
  ASSERT_TRUE(devArrMatch(d_computed_ma_invtrans, d_params, nElements, CompareApprox<double>(params.tolerance)));
  ASSERT_TRUE(devArrMatch(d_computed_ar_invtrans, d_params, nElements, CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(JonesTrans, JonesTransTestClass,
                        ::testing::ValuesIn(inputs));

}  //end namespace TimeSeries
}  //end namespace MLCommon
