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

/// TODO:
///  - test with d > 0, D > 0
///  - more generic, test type (e.g KPSS) as parameter

namespace MLCommon {
namespace TimeSeries {

template <typename DataT>
struct StationarityParams {
  int batch_size;
  int n_samples;
  DataT scale;
  std::vector<DataT> inc_rates;
  std::vector<uint8_t> d_ref;  // vector<bool> is not an array in C++
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
      std::vector<DataT>(params.n_samples * params.batch_size);
    std::vector<DataT> noise = std::vector<DataT>(params.n_samples);
    std::vector<DataT> offset = std::vector<DataT>(params.batch_size);

    constexpr unsigned seed = 12345678U;
    std::mt19937 gen(seed);
    // Generate white noise
    std::normal_distribution<> ndis(0.0, 0.1);
    for (int j = 0; j < params.n_samples; ++j) {
      noise[j] = ndis(gen);
    }
    // Generate an offset for each series
    std::uniform_real_distribution<> udis(-1.0, 1.0);
    for (int i = 0; i < params.batch_size; ++i) {
      offset[i] = udis(gen);
    }

    // Construct the series as a linear signal + offset + noise
    for (int j = 0; j < params.n_samples; j++) {
      x[j] = static_cast<DataT>(j) / n_samples_f;
      for (int i = 0; i < params.batch_size; i++) {
        y[i * params.n_samples + j] =
          params.scale * (x[j] * params.inc_rates[i] + offset[i] + noise[j]);
      }
    }

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocator = std::make_shared<defaultDeviceAllocator>();

    d_y = (DataT*)allocator->allocate(
      params.n_samples * params.batch_size * sizeof(DataT), stream);
    updateDevice(d_y, y.data(), params.n_samples * params.batch_size, stream);

    d_out =
      (bool*)allocator->allocate(params.batch_size * sizeof(bool), stream);

    kpss_test(d_y, d_out, params.batch_size, params.n_samples, 0, 0, 0,
              allocator, stream, static_cast<DataT>(0.05));
  }

  void TearDown() override {
    allocator->deallocate(
      d_y, params.n_samples * params.batch_size * sizeof(DataT), stream);
    allocator->deallocate(d_out, params.batch_size * sizeof(bool), stream);
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  std::shared_ptr<defaultDeviceAllocator> allocator;
  cudaStream_t stream;
  cublasHandle_t cublas_handle;
  StationarityParams<DataT> params;
  DataT* d_y;
  bool* d_out;
};

const std::vector<struct StationarityParams<float>> params_float = {
  {2, 200, 1, {0.5f, 0.0f}, {0, 1}},
    {3, 200, 1, {0.0f, -0.6f, -1.7f}, {1, 0, 0}},
    {3, 241, 1, {-3.7f, 0.7f, 0.0f}, {0, 0, 1}},
    {2, 200, 1234, {2.0f, -3.5f}, {0, 0}}, {
    7, 1000, 442, {0.3f, -1.7f, 0.0f, 0.4f, -0.4f, -4.2f, 1.3f}, {
      0, 0, 1, 0, 0, 0, 0
    }
  }
};

const std::vector<struct StationarityParams<double>> params_double = {
  {5, 1338, 277, {1.0, 0.5, -0.3, 0.0, 2.2}, {0, 0, 0, 1, 0}},
    {2, 500, 1, {0.1, -0.1}, {0, 0}}, {
    17, 104, 99, {-0.5, 0.5,  0.0,  0.0,  0.0, -0.8, -0.6, 0.0, 0.9,
                  0.0,  -1.3, -0.7, -0.4, 1.1, -1.2, -0.6, 0.0},
    {
      0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1
    }
  }
};

typedef StationarityTest<float> StationarityTestF;
TEST_P(StationarityTestF, Result) {
  ASSERT_TRUE(devArrMatchHost((bool*)params.d_ref.data(), d_out,
                              params.batch_size, Compare<bool>(), stream));
}

typedef StationarityTest<double> StationarityTestD;
TEST_P(StationarityTestD, Result) {
  ASSERT_TRUE(devArrMatchHost((bool*)params.d_ref.data(), d_out,
                              params.batch_size, Compare<bool>(), stream));
}

INSTANTIATE_TEST_CASE_P(StationarityTests, StationarityTestF,
                        ::testing::ValuesIn(params_float));

INSTANTIATE_TEST_CASE_P(StationarityTests, StationarityTestD,
                        ::testing::ValuesIn(params_double));

}  //end namespace TimeSeries
}  //end namespace MLCommon
