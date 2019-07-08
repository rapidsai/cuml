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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include "holtwinters/Aion.hpp"
#include "ml_utils.h"

namespace ML {

using namespace MLCommon;

#define AION_SAFE_CALL(call)                                        \
  do {                                                              \
    aion::AionStatus status = call;                                 \
    if (status != aion::AionStatus::AION_SUCCESS) {                 \
      std::cerr << "Aion error in in line " << __LINE__ << std::endl; \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

struct HoltWintersInputs {
  int batch_size;
  int frequency;
  aion::SeasonalType seasonal;
  int start_periods;
};

template <typename T>
class HoltWintersTest : public ::testing::TestWithParam<HoltWintersInputs> {
 public:
  void basicTest() {
    params = ::testing::TestWithParam<HoltWintersInputs>::GetParam();
    batch_size = params.batch_size;
    frequency = params.frequency;
    aion::SeasonalType seasonal = params.seasonal;
    start_periods = params.start_periods;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<T> dataset_h = {
      315.42, 316.31, 316.50, 317.56, 318.13, 318.00, 316.39, 314.65, 313.68,
      313.18, 314.66, 315.43, 316.27, 316.81, 317.42, 318.87, 319.87, 319.43,
      318.01, 315.74, 314.00, 313.68, 314.84, 316.03, 316.73, 317.54, 318.38,
      319.31, 320.42, 319.61, 318.42, 316.63, 314.83, 315.16, 315.94, 316.85,
      317.78, 318.40, 319.53, 320.42, 320.85, 320.45, 319.45, 317.25, 316.11,
      315.27, 316.53, 317.53, 318.58, 318.92, 319.70, 321.22, 322.08, 321.31,
      319.58, 317.61, 316.05, 315.83, 316.91, 318.20, 319.41, 320.07, 320.74,
      321.40, 322.06, 321.73, 320.27, 318.54, 316.54, 316.71, 317.53, 318.55,
      319.27, 320.28, 320.73, 321.97, 322.00, 321.71, 321.05, 318.71, 317.66,
      317.14, 318.70, 319.25, 320.46, 321.43, 322.23, 323.54, 323.91, 323.59,
      322.24, 320.20, 318.48, 317.94, 319.63, 320.87, 322.17, 322.34, 322.88,
      324.25, 324.83, 323.93, 322.38, 320.76, 319.10, 319.24, 320.56, 321.80,
      322.40, 322.99, 323.73, 324.86, 325.40, 325.20, 323.98, 321.95, 320.18,
      320.09, 321.16, 322.74};

    // initial values for alpha, beta and gamma
    std::vector<T> alpha_h(batch_size, 0.4);
    std::vector<T> beta_h(batch_size, 0.3);
    std::vector<T> gamma_h(batch_size, 0.3);

    int leveltrend_seed_len, season_seed_len, components_len;
    int leveltrend_coef_offset, season_coef_offset;
    int error_len;

    AION_SAFE_CALL(aion::HoltWintersBufferSize(
      n, batch_size, frequency, optim_beta, optim_gamma,
      &leveltrend_seed_len,     // = batch_size
      &season_seed_len,         // = frequency*batch_size
      &components_len,          // = (n-w_len)*batch_size
      &error_len,               // = batch_size
      &leveltrend_coef_offset,  // = (n-wlen-1)*batch_size (last row)
      &season_coef_offset));  // = (n-wlen-frequency)*batch_size(last freq rows)

    allocate(dataset_d, batch_size * n);
    allocate(dataset_d_copy, batch_size * n);
    updateDevice(dataset_d_copy, dataset_h.data(), batch_size * n, stream);
    allocate(forecast_d, batch_size * h);
    allocate(alpha_d, batch_size);
    updateDevice(alpha_d, alpha_h.data(), batch_size, stream);
    allocate(level_seed_d, leveltrend_seed_len);
    allocate(level_d, components_len);

    // if optim_beta
    allocate(beta_d, batch_size);
    updateDevice(beta_d, beta_h.data(), batch_size, stream);
    allocate(trend_seed_d, leveltrend_seed_len);
    allocate(trend_d, components_len);

    // if optim_gamma
    allocate(gamma_d, batch_size);
    updateDevice(gamma_d, gamma_h.data(), batch_size, stream);
    allocate(start_season_d, season_seed_len);
    allocate(season_d, components_len);

    allocate(error_d, error_len);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Step 1: transpose the dataset (aion expects col major dataset)
    AION_SAFE_CALL(
      aion::AionTranspose<T>(dataset_d_copy, batch_size, n, dataset_d, mode));

    // Step 2: Decompose dataset to get seed for level, trend and seasonal values
    AION_SAFE_CALL(aion::HoltWintersDecompose<T>(
      dataset_d, n, batch_size, frequency, level_seed_d, trend_seed_d,
      start_season_d, start_periods, seasonal, mode));

    // Step 3: Find optimal alpha, beta and gamma values (seasonal HW)
    AION_SAFE_CALL(aion::HoltWintersOptim<T>(
      dataset_d, n, batch_size, frequency, level_seed_d, trend_seed_d,
      start_season_d, alpha_d, optim_alpha, beta_d, optim_beta, gamma_d,
      optim_gamma, level_d, trend_d, season_d, nullptr, error_d, nullptr,
      nullptr, seasonal, mode));

    // Step 4: Do forecast
    AION_SAFE_CALL(aion::HoltWintersForecast<T>(
      forecast_d, h, batch_size, frequency, level_d + leveltrend_coef_offset,
      trend_d + leveltrend_coef_offset, season_d + season_coef_offset, seasonal,
      mode));
  }

  void SetUp() override {
    AION_SAFE_CALL(aion::AionInit());
    basicTest();
  }

  void TearDown() override {
    AION_SAFE_CALL(aion::AionDestroy());
    CUDA_CHECK(cudaFree(dataset_d));
    CUDA_CHECK(cudaFree(dataset_d_copy));
    CUDA_CHECK(cudaFree(forecast_d));
    CUDA_CHECK(cudaFree(level_seed_d));
    CUDA_CHECK(cudaFree(trend_seed_d));
    CUDA_CHECK(cudaFree(start_season_d));
    CUDA_CHECK(cudaFree(level_d));
    CUDA_CHECK(cudaFree(trend_d));
    CUDA_CHECK(cudaFree(season_d));
    CUDA_CHECK(cudaFree(alpha_d));
    CUDA_CHECK(cudaFree(beta_d));
    CUDA_CHECK(cudaFree(gamma_d));
    CUDA_CHECK(cudaFree(error_d));
  }

 public:
  HoltWintersInputs params;
  T *dataset_d, *dataset_d_copy;
  T *forecast_d;
  T *level_seed_d, *trend_seed_d = nullptr, *start_season_d = nullptr;
  T *level_d, *trend_d = nullptr, *season_d = nullptr;
  T *alpha_d, *beta_d = nullptr, *gamma_d = nullptr;
  T *error_d;
  int n = 120, h = 50;
  int batch_size, frequency, start_periods;
  bool optim_alpha = true, optim_beta = true, optim_gamma = true;
  aion::ComputeMode mode = aion::ComputeMode::GPU;
};

const std::vector<HoltWintersInputs> inputsf = {
  {1, 12, aion::SeasonalType::ADDITIVE, 2}};

typedef HoltWintersTest<float> HoltWintersTestF;
TEST_P(HoltWintersTestF, Fit) {
  myPrintDevVector("alpha", (const float *)alpha_d, batch_size);
  myPrintDevVector("beta", (const float *)beta_d, batch_size);
  myPrintDevVector("gamma", (const float *)gamma_d, batch_size);
  myPrintDevVector("forecast", (const float *)forecast_d, h);
  myPrintDevVector("error", (const float *)error_d, batch_size);
  ASSERT_TRUE(true == true);
}

INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestF,
                        ::testing::ValuesIn(inputsf));

}  // namespace ML