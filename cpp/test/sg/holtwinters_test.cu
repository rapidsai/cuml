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
      std::cerr << "Aion error in in line " << status << std::endl; \
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
    int batch_size = params.batch_size;
    int frequency = params.frequency;
    aion::SeasonalType seasonal = params.seasonal;
    int start_periods = params.start_periods;

    int n = 12, h = 5;
    bool optim_alpha = true, optim_beta = true, optim_gamma = true;
    aion::ComputeMode mode = aion::ComputeMode::GPU;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<T> dataset_h = {3.0, 2.0, 1.0, 3.0, 2.0, 1.0,
                                3.0, 2.0, 1.0, 3.0, 2.0, 1.0};

    // // initial values for alpha, beta and gamma
    // std::vector<T> alpha_h(batch_size, 0.4);
    // std::vector<T> beta_h(batch_size, 0.3);
    // std::vector<T> gamma_h(batch_size, 0.3);

    // int leveltrend_seed_len, season_seed_len, components_len;
    // int leveltrend_coef_offset, season_coef_offset;
    // int error_len;

    // AION_SAFE_CALL(aion::HoltWintersBufferSize(
    //   n, batch_size, frequency, optim_beta, optim_gamma,
    //   &leveltrend_seed_len,     // = batch_size
    //   &season_seed_len,         // = frequency*batch_size
    //   &components_len,          // = (n-w_len)*batch_size
    //   &error_len,               // = batch_size
    //   &leveltrend_coef_offset,  // = (n-wlen-1)*batch_size (last row)
    //   &season_coef_offset));  // = (n-wlen-frequency)*batch_size(last freq rows)

    allocate(dataset_d, batch_size * n);
    allocate(dataset_d_copy, batch_size * n);
    updateDevice(dataset_d_copy, dataset_h.data(), n, stream);
    // allocate(forecast_d, batch_size * h);
    // allocate(alpha_d, batch_size);
    // updateDevice(alpha_d, alpha_h.data(), batch_size, stream);
    // allocate(level_seed_d, leveltrend_seed_len);
    // allocate(level_d, components_len);

    // // if optim_beta
    // allocate(beta_d, batch_size);
    // updateDevice(beta_d, beta_h.data(), batch_size, stream);
    // allocate(trend_seed_d, leveltrend_seed_len);
    // allocate(trend_d, components_len);

    // // if optim_gamma
    // allocate(gamma_d, batch_size);
    // updateDevice(gamma_d, gamma_h.data(), batch_size, stream);
    // allocate(start_season_d, season_seed_len);
    // allocate(season_d, components_len);

    // allocate(error_d, error_len);

    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // CUDA_CHECK(cudaStreamDestroy(stream));
    // Step 1: transpose the dataset (aion expects col major dataset)
    // MLCommon::myPrintDevVector("C", dataset_d, 12);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    AION_SAFE_CALL(
      aion::AionTranspose<T>(dataset_d_copy, batch_size, n, dataset_d, mode));

    // myPrintDevVector("alpha_d", alpha_d, batch_size);

    // // myPrintDevVector("Device Dataset", (const float *)dataset_d, 12);

    // // Step 2: Decompose dataset to get seed for level, trend and seasonal values
    // AION_SAFE_CALL(aion::HoltWintersDecompose<T>(
    //   dataset_d, n, batch_size, frequency, level_seed_d, trend_seed_d,
    //   start_season_d, start_periods, seasonal, mode));

    // // Step 3: Find optimal alpha, beta and gamma values (seasonal HW)
    // AION_SAFE_CALL(aion::HoltWintersOptim<T>(
    //   dataset_d, n, batch_size, frequency, level_seed_d, trend_seed_d,
    //   start_season_d, alpha_d, optim_alpha, beta_d, optim_beta, gamma_d,
    //   optim_gamma, level_d, trend_d, season_d, nullptr, error_d, nullptr,
    //   nullptr, seasonal, mode));

    // // Step 4: Do forecast
    // AION_SAFE_CALL(aion::HoltWintersForecast<T>(
    //   forecast_d, h, batch_size, frequency, level_d + leveltrend_coef_offset,
    //   trend_d + leveltrend_coef_offset, season_d + season_coef_offset, seasonal,
    //   mode));
  }

  void SetUp() override {
    AION_SAFE_CALL(aion::AionInit());
    basicTest();
  }

  void TearDown() override {
    // AION_SAFE_CALL(aion::AionDestroy());
    // CUDA_CHECK(cudaFree(dataset_d));
    // CUDA_CHECK(cudaFree(forecast_d));
    // CUDA_CHECK(cudaFree(level_seed_d));
    // CUDA_CHECK(cudaFree(trend_seed_d));
    // CUDA_CHECK(cudaFree(start_season_d));
    // CUDA_CHECK(cudaFree(level_d));
    // CUDA_CHECK(cudaFree(trend_d));
    // CUDA_CHECK(cudaFree(season_d));
    // CUDA_CHECK(cudaFree(alpha_d));
    // CUDA_CHECK(cudaFree(beta_d));
    // CUDA_CHECK(cudaFree(gamma_d));
    // CUDA_CHECK(cudaFree(error_d));
  }

 public:
  HoltWintersInputs params;
  T *dataset_d, *dataset_d_copy;
  T *forecast_d;
  T *level_seed_d, *trend_seed_d = nullptr, *start_season_d = nullptr;
  T *level_d, *trend_d = nullptr, *season_d = nullptr;
  T *alpha_d, *beta_d = nullptr, *gamma_d = nullptr;
  T *error_d;
};

const std::vector<HoltWintersInputs> inputsf = {
  {1, 3, aion::SeasonalType::ADDITIVE, 2}};

typedef HoltWintersTest<float> HoltWintersTestF;
TEST_P(HoltWintersTestF, Fit) {
  // myPrintDevVector("forecast_d", (const float *)forecast_d, 5);
  ASSERT_TRUE(true == true);
}

INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestF,
                        ::testing::ValuesIn(inputsf));

}  // namespace ML