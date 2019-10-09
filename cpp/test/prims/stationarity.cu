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
#include <iostream>
#include <random>
#include <vector>
#include "test_utils.h"
#include "timeSeries/stationarity.cuh"

/* /!\ TODO: test non stationary case? (error) /!\ */

namespace MLCommon {
namespace TimeSeries {

template <typename DataT>
struct StationarityParams {
  size_t n_batches;
  size_t n_samples;
  DataT scale;
  std::vector<DataT> inc_rates;
  std::vector<int> d_ref;
};

template <typename DataT>
class StationarityTest
  : public ::testing::TestWithParam<StationarityParams<DataT>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<StationarityParams<DataT>>::GetParam();

    DataT n_samples_f = static_cast<float>(params.n_samples);

    std::vector<DataT> x = std::vector<DataT>(params.n_samples);
    std::vector<DataT> y =
      std::vector<DataT>(params.n_samples * params.n_batches);
    std::vector<DataT> noise = std::vector<DataT>(params.n_samples);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 0.1);
    for (int j = 0; j < params.n_samples; ++j) {
      noise[j] = dis(gen);
    }

    for (int j = 0; j < params.n_samples; j++) {
      x[j] = static_cast<DataT>(j) / n_samples_f;
      for (int i = 0; i < params.n_batches; i++) {
        y[i * params.n_samples + j] = x[j] * params.inc_rates[i] + noise[j];
      }
    }

    d_out = std::vector<int>(params.n_batches);

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    MLCommon::TimeSeries::stationarity(y.data(), d_out.data(), params.n_batches,
                                       params.n_samples, stream, cublas_handle,
                                       static_cast<DataT>(0.05));
  }

  void TearDown() override {
    // TODO: free when using device code
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream;
  cublasHandle_t cublas_handle;
  StationarityParams<DataT> params;
  std::vector<int> d_out;
};

// TODO: remove me
template <typename DataT, typename F>
::testing::AssertionResult arrMatch(const DataT *expected_h,
                                    const DataT *actual_h, size_t size,
                                    F eq_compare) {
  bool ok = true;
  auto fail = ::testing::AssertionFailure();
  for (size_t i(0); i < size; ++i) {
    auto exp = expected_h[i];
    auto act = actual_h[i];
    if (!eq_compare(exp, act)) {
      ok = false;
      fail << "actual=" << act << " != expected=" << exp << " @" << i << "; ";
    }
  }
  if (!ok) return fail;
  return ::testing::AssertionSuccess();
}

/* Setting test parameter values */
const std::vector<struct StationarityParams<float>> params_float = {
  /*   {2, 200, 1, {0.5f, 0.0f}, {1, 0}},               // basic test
    {2, 200, 1, {0.0f, -0.2f, -1.7f}, {0, 1, 1}},  // some decreasing series
    {2, 200, 1234, {2.0f, -3.5f}, {1, 1}},         // larger values
  {
    7, 1000, 442, {0.3f, -1.7f, 0.0f, 0.4f, -0.2f, -4.2f, 1.3f}, {
      1, 1, 0, 1, 1, 1, 1
    }
  }  // multiple large series */
  {
    2, 241, 1, {0.2f, -0.2f}, { 1, 1 }  // try to break alignment
  }
};

// TODO: non-even n_samples

const std::vector<struct StationarityParams<double>> params_double = {
  /*   {5, 1338, 277, {1.0f, 0.5f, -0.3f, 0.0f, 2.2f}, {1, 1, 1, 0, 1}},
  // multiple large series
  {
    2, 500, 1, {0.1f, -0.1f}, { 1, 1 }
  }  // almost stationary series */
  {
    2, 241, 1, {0.2f, -0.2f}, { 1, 1 }  // try to break alignment
  }
};

typedef StationarityTest<float> StationarityTestF;
TEST_P(StationarityTestF, Result) {
  ASSERT_TRUE(arrMatch(params.d_ref.data(), d_out.data(), params.n_batches,
                       Compare<int>()));
  // TODO: device -> switch to devArrMatch
}

typedef StationarityTest<double> StationarityTestD;
TEST_P(StationarityTestD, Result) {
  ASSERT_TRUE(arrMatch(params.d_ref.data(), d_out.data(), params.n_batches,
                       Compare<int>()));
  // TODO: device -> switch to devArrMatch
}

INSTANTIATE_TEST_CASE_P(StationarityTests, StationarityTestF,
                        ::testing::ValuesIn(params_float));

INSTANTIATE_TEST_CASE_P(StationarityTests, StationarityTestD,
                        ::testing::ValuesIn(params_double));

}  //end namespace TimeSeries
}  //end namespace MLCommon
