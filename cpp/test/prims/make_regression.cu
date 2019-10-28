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

#include "cuda_utils.h"
#include "random/make_regression.h"
#include "test_utils.h"

namespace MLCommon {
namespace Random {

template <typename T>
struct MakeRegressionInputs {
  T tolerance;
  int n_samples, n_features, n_informative, n_targets, effective_rank;
  T bias;
  bool shuffle;
  GeneratorType gtype;
  uint64_t seed;
};

template <typename T>
class MakeRegressionTest
  : public ::testing::TestWithParam<MakeRegressionInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MakeBlobsInputs<T>>::GetParam();
    T noise = (T)0.0, tail_strength = (T)0.5;

    allocator.reset(new defaultDeviceAllocator);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocate(data, params.n_samples * params.n_features);
    allocate(values, params.n_samples * params.n_targets);

    // Create the regression problem
    make_regression(data, values, params.n_samples, params.n_features,
                    params.n_informative, cublas_handle, cusolver_handle,
                    allocator, stream, params.n_targets, params.bias,
                    params.effective_rank, tail_strength, noise,
                    params.shuffle, params.seed, params.gtype);
    
    // TODO: option to return coeff so we can test the feature correctly
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(values));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  MakeRegressionInputs<T> params;
  T *data, *values;
  std::shared_ptr<deviceAllocator> allocator;
  cudaStream_t stream;
  cublasHandle_t cublas_handle;
  cusolverDnHandle_t cusolver_handle;
};

typedef MakeRegressionTest<float> MakeRegressionTestF;
const std::vector<MakeRegressionInputs<float>> inputsf_t = {};

TEST_P(MakeRegressionTestF, Result) {}
INSTANTIATE_TEST_CASE_P(MakeRegressionTests, MakeRegressionTestF,
                        ::testing::ValuesIn(inputsf_t));

typedef MakeRegressionTest<double> MakeRegressionTestD;
const std::vector<MakeRegressionInputs<double>> inputsd_t = {};

TEST_P(MakeRegressionTestD, Result) {}
INSTANTIATE_TEST_CASE_P(MakeRegressionTests, MakeRegressionTestD,
                        ::testing::ValuesIn(inputsd_t));

}  // end namespace Random
}  // end namespace MLCommon
