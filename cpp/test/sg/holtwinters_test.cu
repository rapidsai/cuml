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
#include <algorithm>
#include "holtwinters/holtwinters.h"
#include "time_series_datasets.h"

namespace ML {

using namespace MLCommon;

template <typename T>
struct HoltWintersInputs {
  T *dataset_h;
  T *test;
  int n;
  int h;
  int batch_size;
  int frequency;
  ML::SeasonalType seasonal;
  int start_periods;
};

template <typename T>
class HoltWintersTest : public ::testing::TestWithParam<HoltWintersInputs<T>> {
 public:
  void basicTest() {
    params = ::testing::TestWithParam<HoltWintersInputs<T>>::GetParam();
    dataset_h = params.dataset_h;
    test = params.test;
    n = params.n;
    h = params.h;
    batch_size = params.batch_size;
    frequency = params.frequency;
    ML::SeasonalType seasonal = params.seasonal;
    start_periods = params.start_periods;

    CUDA_CHECK(cudaStreamCreate(&stream));

    ML::HoltWinters::buffer_size(
      n, batch_size, frequency,
      &leveltrend_seed_len,     // = batch_size
      &season_seed_len,         // = frequency*batch_size
      &components_len,          // = (n-w_len)*batch_size
      &error_len,               // = batch_size
      &leveltrend_coef_offset,  // = (n-wlen-1)*batch_size (last row)
      &season_coef_offset);  // = (n-wlen-frequency)*batch_size(last freq rows)

    allocate(level_ptr, components_len, stream);
    allocate(trend_ptr, components_len, stream);
    allocate(season_ptr, components_len, stream);
    allocate(SSE_error_ptr, batch_size, stream);
    allocate(forecast_ptr, batch_size * h, stream);

    allocate(data, batch_size * n);
    updateDevice(data, dataset_h, batch_size * n, stream);

    cumlHandle handle;
    handle.setStream(stream);

    ML::HoltWinters::fit(handle, n, batch_size, frequency, start_periods,
                         seasonal, data, level_ptr, trend_ptr, season_ptr,
                         SSE_error_ptr);

    ML::HoltWinters::forecast(handle, n, batch_size, frequency, h, seasonal,
                              level_ptr, trend_ptr, season_ptr, forecast_ptr);

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(level_ptr));
    CUDA_CHECK(cudaFree(trend_ptr));
    CUDA_CHECK(cudaFree(season_ptr));
    CUDA_CHECK(cudaFree(SSE_error_ptr));
    CUDA_CHECK(cudaFree(forecast_ptr));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 public:
  cudaStream_t stream;
  HoltWintersInputs<T> params;
  T *dataset_h, *test;
  T *data;
  int n, h;
  int leveltrend_seed_len, season_seed_len, components_len;
  int leveltrend_coef_offset, season_coef_offset;
  int error_len;
  int batch_size, frequency, start_periods;
  T *SSE_error_ptr, *level_ptr, *trend_ptr, *season_ptr, *forecast_ptr;
};

const std::vector<HoltWintersInputs<float>> inputsf = {
  {additive_trainf.data(), additive_testf.data(), 90, 10, 1, 25,
   ML::SeasonalType::ADDITIVE, 2},
  {multiplicative_trainf.data(), multiplicative_testf.data(), 132, 12, 1, 12,
   ML::SeasonalType::MULTIPLICATIVE, 2},
  {additive_normalized_trainf.data(), additive_normalized_testf.data(), 90, 10,
   1, 25, ML::SeasonalType::ADDITIVE, 2},
  {multiplicative_normalized_trainf.data(),
   multiplicative_normalized_testf.data(), 132, 12, 1, 12,
   ML::SeasonalType::MULTIPLICATIVE, 2}};

const std::vector<HoltWintersInputs<double>> inputsd = {
  {additive_traind.data(), additive_testd.data(), 90, 10, 1, 25,
   ML::SeasonalType::ADDITIVE, 2},
  {multiplicative_traind.data(), multiplicative_testd.data(), 132, 12, 1, 12,
   ML::SeasonalType::MULTIPLICATIVE, 2},
  {additive_normalized_traind.data(), additive_normalized_testd.data(), 90, 10,
   1, 25, ML::SeasonalType::ADDITIVE, 2},
  {multiplicative_normalized_traind.data(),
   multiplicative_normalized_testd.data(), 132, 12, 1, 12,
   ML::SeasonalType::MULTIPLICATIVE, 2}};

template <typename T>
void normalise(T *data, int len) {
  T min = *std::min_element(data, data + len);
  T max = *std::max_element(data, data + len);
  for (int i = 0; i < len; i++) {
    data[i] = (data[i] - min) / (max - min);
  }
}

template <typename T>
T calculate_MAE(T *test, T *forecast, int batch_size, int h) {
  normalise(test, batch_size * h);
  normalise(forecast, batch_size * h);
  std::vector<T> ae(batch_size * h);
  for (int i = 0; i < batch_size * h; i++) {
    ae[i] = abs(test[i] - forecast[i]);
  }
  std::sort(ae.begin(), ae.end());
  T mae;
  if (h % 2 == 0) {
    mae = (ae[h / 2 - 1] + ae[h / 2]) / 2;
  } else {
    mae = ae[(int)h / 2];
  }
  return mae;
}

typedef HoltWintersTest<float> HoltWintersTestF;
TEST_P(HoltWintersTestF, Fit) {
  std::vector<float> forecast_h(batch_size * h);
  updateHost(forecast_h.data(), forecast_ptr, batch_size * h, stream);
  myPrintHostVector("forecast", forecast_h.data(), batch_size * h);
  float mae = calculate_MAE<float>(test, forecast_h.data(), batch_size, h);
  std::cout << "MAE: " << mae << std::endl;
}

typedef HoltWintersTest<double> HoltWintersTestD;
TEST_P(HoltWintersTestD, Fit) {
  std::vector<double> forecast_h(batch_size * h);
  updateHost(forecast_h.data(), forecast_ptr, batch_size * h, stream);
  myPrintHostVector("forecast", forecast_h.data(), batch_size * h);
  double mae = calculate_MAE<double>(test, forecast_h.data(), batch_size, h);
  std::cout << "MAE: " << mae << std::endl;
}

INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestF,
                        ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestD,
                        ::testing::ValuesIn(inputsd));

}  // namespace ML