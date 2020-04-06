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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "cuml/common/cuml_allocator.hpp"
#include "test_utils.h"
#include "timeSeries/stationarity.h"

namespace MLCommon {
namespace TimeSeries {

template <typename DataT>
struct StationarityParams {
  int n_batches;
  int n_samples;
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
    std::vector<DataT> offset = std::vector<DataT>(params.n_batches);

    constexpr unsigned seed = 12345678U;
    std::mt19937 gen(seed);
    // Generate white noise
    std::normal_distribution<> ndis(0.0, 0.1);
    for (int j = 0; j < params.n_samples; ++j) {
      noise[j] = ndis(gen);
    }
    // Generate an offset for each series
    std::uniform_real_distribution<> udis(-1.0, 1.0);
    for (int i = 0; i < params.n_batches; ++i) {
      offset[i] = udis(gen);
    }

    // Construct the series as a linear signal + offset + noise
    for (int j = 0; j < params.n_samples; j++) {
      x[j] = static_cast<DataT>(j) / n_samples_f;
      for (int i = 0; i < params.n_batches; i++) {
        y[i * params.n_samples + j] =
          params.scale * (x[j] * params.inc_rates[i] + offset[i] + noise[j]);
      }
    }

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUDA_CHECK(cudaStreamCreate(&stream));

    MLCommon::allocate(y_d, params.n_samples * params.n_batches);
    MLCommon::updateDevice(y_d, y.data(), params.n_samples * params.n_batches,
                           stream);

    std::shared_ptr<MLCommon::deviceAllocator> allocator(
      new defaultDeviceAllocator);

    d_out = std::vector<int>(params.n_batches);

    MLCommon::TimeSeries::stationarity(y_d, d_out.data(), params.n_batches,
                                       params.n_samples, allocator, stream,
                                       static_cast<DataT>(0.05));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(y_d));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream;
  cublasHandle_t cublas_handle;
  StationarityParams<DataT> params;
  DataT *y_d;
  std::vector<int> d_out;
};

/* The tests respectively check the following aspects:
 *  - basic test with trends 0 and 1
 *  - some decreasing series
 *  - odd series size
 *  - larger values
 *  - multiple large series
 */
const std::vector<struct StationarityParams<float>> params_float = {
  {2, 200, 1, {0.5f, 0.0f}, {1, 0}},
    {3, 200, 1, {0.0f, -0.6f, -1.7f}, {0, 1, 1}},
    {3, 241, 1, {-3.7f, 0.7f, 0.0f}, {1, 1, 0}},
    {2, 200, 1234, {2.0f, -3.5f}, {1, 1}}, {
    7, 1000, 442, {0.3f, -1.7f, 0.0f, 0.4f, -0.4f, -4.2f, 1.3f}, {
      1, 1, 0, 1, 1, 1, 1
    }
  }
};

/* The tests respectively check the following aspects:
 *  - multiple large series
 *  - almost stationary series
 *  - many small series
 */
const std::vector<struct StationarityParams<double>> params_double = {
  {5, 1338, 277, {1.0, 0.5, -0.3, 0.0, 2.2}, {1, 1, 1, 0, 1}},
    {2, 500, 1, {0.1, -0.1}, {1, 1}}, {
    17, 104, 99, {-0.5, 0.5,  0.0,  0.0,  0.0, -0.8, -0.6, 0.0, 0.9,
                  0.0,  -1.3, -0.7, -0.4, 1.1, -1.2, -0.6, 0.0},
    {
      1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0
    }
  }
};

typedef StationarityTest<float> StationarityTestF;
TEST_P(StationarityTestF, Result) {
  ASSERT_THAT(d_out, ::testing::ElementsAreArray(params.d_ref));
}

typedef StationarityTest<double> StationarityTestD;
TEST_P(StationarityTestD, Result) {
  ASSERT_THAT(d_out, ::testing::ElementsAreArray(params.d_ref));
}

INSTANTIATE_TEST_CASE_P(StationarityTests, StationarityTestF,
                        ::testing::ValuesIn(params_float));

INSTANTIATE_TEST_CASE_P(StationarityTests, StationarityTestD,
                        ::testing::ValuesIn(params_double));

}  //end namespace TimeSeries
}  //end namespace MLCommon
