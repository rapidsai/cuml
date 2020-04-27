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
#include <thrust/count.h>
#include <thrust/device_vector.h>

#include "cuda_utils.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/subtract.h"
#include "linalg/transpose.h"
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
    params = ::testing::TestWithParam<MakeRegressionInputs<T>>::GetParam();

    // Noise must be zero to compare the actual and expected values
    T noise = (T)0.0, tail_strength = (T)0.5;

    allocator.reset(new defaultDeviceAllocator);
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocate(data, params.n_samples * params.n_features);
    allocate(values_ret, params.n_samples * params.n_targets);
    allocate(values_prod, params.n_samples * params.n_targets);
    allocate(values_cm, params.n_samples * params.n_targets);
    allocate(coef, params.n_features * params.n_targets);

    // Create the regression problem
    make_regression(data, values_ret, params.n_samples, params.n_features,
                    params.n_informative, cublas_handle, cusolver_handle,
                    allocator, stream, coef, params.n_targets, params.bias,
                    params.effective_rank, tail_strength, noise, params.shuffle,
                    params.seed, params.gtype);

    // Calculate the values from the data and coefficients (column-major)
    T alpha = (T)1.0, beta = (T)0.0;
    CUBLAS_CHECK(LinAlg::cublasgemm(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, params.n_samples,
      params.n_targets, params.n_features, &alpha, data, params.n_features,
      coef, params.n_targets, &beta, values_cm, params.n_samples, stream));

    // Transpose the values to row-major
    LinAlg::transpose(values_cm, values_prod, params.n_samples,
                      params.n_targets, cublas_handle, stream);

    // Add the bias
    LinAlg::addScalar(values_prod, values_prod, params.bias,
                      params.n_samples * params.n_targets, stream);

    // Count the number of zeroes in the coefficients
    thrust::device_ptr<T> __coef = thrust::device_pointer_cast(coef);
    zero_count = thrust::count(
      __coef, __coef + params.n_features * params.n_targets, (T)0.0);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(values_ret));
    CUDA_CHECK(cudaFree(values_prod));
    CUDA_CHECK(cudaFree(values_cm));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  MakeRegressionInputs<T> params;
  T *data, *values_ret, *values_prod, *values_cm, *coef;
  int zero_count;
  std::shared_ptr<deviceAllocator> allocator;
  cudaStream_t stream;
  cublasHandle_t cublas_handle;
  cusolverDnHandle_t cusolver_handle;
};

typedef MakeRegressionTest<float> MakeRegressionTestF;
const std::vector<MakeRegressionInputs<float>> inputsf_t = {
  {0.01f, 256, 32, 16, 1, -1, 0.f, true, GenPhilox, 1234ULL},
  {0.01f, 1000, 100, 47, 4, 65, 4.2f, true, GenPhilox, 1234ULL},
  {0.01f, 20000, 500, 450, 13, -1, -3.f, false, GenPhilox, 1234ULL}};

TEST_P(MakeRegressionTestF, Result) {
  ASSERT_TRUE(
    match(params.n_targets * (params.n_features - params.n_informative),
          zero_count, Compare<int>()));
  ASSERT_TRUE(devArrMatch(values_ret, values_prod, params.n_samples,
                          params.n_targets,
                          CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(MakeRegressionTests, MakeRegressionTestF,
                        ::testing::ValuesIn(inputsf_t));

typedef MakeRegressionTest<double> MakeRegressionTestD;
const std::vector<MakeRegressionInputs<double>> inputsd_t = {
  {0.01, 256, 32, 16, 1, -1, 0.0, true, GenPhilox, 1234ULL},
  {0.01, 1000, 100, 47, 4, 65, 4.2, true, GenPhilox, 1234ULL},
  {0.01, 20000, 500, 450, 13, -1, -3.0, false, GenPhilox, 1234ULL}};

TEST_P(MakeRegressionTestD, Result) {
  ASSERT_TRUE(
    match(params.n_targets * (params.n_features - params.n_informative),
          zero_count, Compare<int>()));
  ASSERT_TRUE(devArrMatch(values_ret, values_prod, params.n_samples,
                          params.n_targets,
                          CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(MakeRegressionTests, MakeRegressionTestD,
                        ::testing::ValuesIn(inputsd_t));

}  // end namespace Random
}  // end namespace MLCommon
