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
#include "holtwinters/holtwinters.h"

namespace ML {

using namespace MLCommon;

struct HoltWintersInputs {
  int n;
  int h;
  int batch_size;
  int frequency;
  ML::SeasonalType seasonal;
  int start_periods;
};

template <typename T>
class HoltWintersTest : public ::testing::TestWithParam<HoltWintersInputs> {
 public:
  void basicTest() {
    params = ::testing::TestWithParam<HoltWintersInputs>::GetParam();
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

    std::vector<T> dataset_h;
    if (seasonal == ML::SeasonalType::ADDITIVE) {
      dataset_h = {
        0.5,        0.6245908,  0.74135309, 0.84295029, 0.92299865, 0.97646846,
        1.,         0.9921147,  0.95330803, 0.88601834, 0.7944737,  0.6844262,
        0.56279052, 0.43720948, 0.3155738,  0.2055263,  0.11398166, 0.04669197,
        0.0078853,  0.,         0.02353154, 0.07700135, 0.15704971, 0.25864691,
        0.3754092,  0.5,        0.6245908,  0.74135309, 0.84295029, 0.92299865,
        0.97646846, 1.,         0.9921147,  0.95330803, 0.88601834, 0.7944737,
        0.6844262,  0.56279052, 0.43720948, 0.3155738,  0.2055263,  0.11398166,
        0.04669197, 0.0078853,  0.,         0.02353154, 0.07700135, 0.15704971,
        0.25864691, 0.3754092,  0.5,        0.6245908,  0.74135309, 0.84295029,
        0.92299865, 0.97646846, 1.,         0.9921147,  0.95330803, 0.88601834,
        0.7944737,  0.6844262,  0.56279052, 0.43720948, 0.3155738,  0.2055263,
        0.11398166, 0.04669197, 0.0078853,  0.,         0.02353154, 0.07700135,
        0.15704971, 0.25864691, 0.3754092,  0.5,        0.6245908,  0.74135309,
        0.84295029, 0.92299865, 0.97646846, 1.,         0.9921147,  0.95330803,
        0.88601834, 0.7944737,  0.6844262,  0.56279052, 0.43720948, 0.3155738};
    } else {
      dataset_h = {
        0.01644402, 0.02802703, 0.05505405, 0.04926255, 0.03381853, 0.06084556,
        0.08594208, 0.08594208, 0.06277606, 0.02995753, 0.001,      0.02802703,
        0.02223552, 0.04347104, 0.07242857, 0.06084556, 0.04154054, 0.08787259,
        0.12841313, 0.12841313, 0.1052471,  0.05698456, 0.02030502, 0.07049807,
        0.08015058, 0.08980309, 0.14385714, 0.11489961, 0.13227413, 0.14385714,
        0.18439768, 0.18439768, 0.15544015, 0.11296911, 0.08208108, 0.12069112,
        0.13034363, 0.14771815, 0.17281467, 0.14964865, 0.15350965, 0.22107722,
        0.24424324, 0.26740927, 0.2037027,  0.16895367, 0.13227413, 0.17474517,
        0.17860618, 0.17860618, 0.25582625, 0.25389575, 0.24231274, 0.26933977,
        0.30988031, 0.32532432, 0.25775676, 0.20756371, 0.14771815, 0.18825869,
        0.19405019, 0.16316216, 0.25389575, 0.23845174, 0.25196525, 0.30988031,
        0.38323938, 0.36586486, 0.3002278,  0.24231274, 0.19211969, 0.24231274,
        0.26740927, 0.25003475, 0.31567181, 0.31953282, 0.32146332, 0.40833591,
        0.5029305,  0.47011197, 0.4025444,  0.32918533, 0.25775676, 0.33690734,
        0.34849035, 0.33497683, 0.41219691, 0.4044749,  0.41412741, 0.52223552,
        0.5975251,  0.58208108, 0.48555598, 0.39096139, 0.32339382, 0.39096139,
        0.40833591, 0.38130888, 0.48748649, 0.47204247, 0.48555598, 0.61489961,
        0.6979112,  0.7017722,  0.58015058, 0.47011197, 0.38903089, 0.44887645,
        0.45659846, 0.41412741, 0.4990695,  0.47204247, 0.501,      0.63999614,
        0.74810425, 0.77513127, 0.58015058, 0.49327799, 0.3986834,  0.45080695,
        0.49520849, 0.46045946, 0.58401158, 0.56470656, 0.61103861, 0.71142471,
        0.85814286, 0.87937838, 0.69405019, 0.58594208, 0.4990695,  0.58208108};
    }

    allocate(data, batch_size * n);
    updateDevice(data, dataset_h.data(), batch_size * n, stream);

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
  HoltWintersInputs params;
  T *data;
  int n, h;
  int leveltrend_seed_len, season_seed_len, components_len;
  int leveltrend_coef_offset, season_coef_offset;
  int error_len;
  int batch_size, frequency, start_periods;
  double *SSE_error_ptr;
  double *level_ptr, *trend_ptr, *season_ptr;
  Dtype *forecast_ptr;
};

const std::vector<HoltWintersInputs> inputsf1 = {
  {90, 10, 1, 25, ML::SeasonalType::ADDITIVE, 2}};
const std::vector<HoltWintersInputs> inputsf2 = {
  {132, 12, 1, 12, ML::SeasonalType::MULTIPLICATIVE, 2}};
const std::vector<HoltWintersInputs> inputsd2 = {
  {132, 12, 1, 12, ML::SeasonalType::MULTIPLICATIVE, 2}};

typedef HoltWintersTest<float> HoltWintersTestAF;
TEST_P(HoltWintersTestAF, Fit) {
  std::vector<float> test = {0.2055263,  0.11398166, 0.04669197, 0.0078853,
                             0.,         0.02353154, 0.07700135, 0.15704971,
                             0.25864691, 0.3754092};
  std::vector<float> forecast_h(batch_size * h);
  updateHost(forecast_h.data(), forecast_ptr, batch_size * h, stream);
  myPrintHostVector("forecast", forecast_h.data(), batch_size * h);
  std::vector<float> ae(batch_size * h);
  for (int i = 0; i < batch_size * h; i++) {
    ae[i] = abs(test[i] - forecast_h[i]);
  }
  std::sort(ae.begin(), ae.end());
  float mae;
  if (h % 2 == 0) {
    mae = (ae[h / 2 - 1] + ae[h / 2]) / 2;
  } else {
    mae = ae[(int)h / 2];
  }
  std::cout << "MAE: " << mae << std::endl;
}

typedef HoltWintersTest<float> HoltWintersTestMF;
TEST_P(HoltWintersTestMF, Fit) {
  std::vector<float> test = {0.6052471,  0.55505405, 0.60910811, 0.69018919,
                             0.71142471, 0.83304633, 1.001,      0.97011197,
                             0.78092278, 0.69018919, 0.55312355, 0.63420463};
  std::vector<float> forecast_h(batch_size * h);
  updateHost(forecast_h.data(), forecast_ptr, batch_size * h, stream);
  myPrintHostVector("forecast", forecast_h.data(), batch_size * h);
  std::vector<float> ae(batch_size * h);
  for (int i = 0; i < batch_size * h; i++) {
    ae[i] = abs(test[i] - forecast_h[i]);
  }
  std::sort(ae.begin(), ae.end());
  float mae;
  if (h % 2 == 0) {
    mae = (ae[h / 2 - 1] + ae[h / 2]) / 2;
  } else {
    mae = ae[(int)h / 2];
  }
  std::cout << "MAE: " << mae << std::endl;
}

typedef HoltWintersTest<double> HoltWintersTestMD;
TEST_P(HoltWintersTestMD, Fit) {
  std::vector<double> test = {0.6052471,  0.55505405, 0.60910811, 0.69018919,
                              0.71142471, 0.83304633, 1.001,      0.97011197,
                              0.78092278, 0.69018919, 0.55312355, 0.63420463};
  std::vector<double> forecast_h(batch_size * h);
  updateHost(forecast_h.data(), forecast_ptr, batch_size * h, stream);
  myPrintHostVector("forecast", forecast_h.data(), batch_size * h);
  std::vector<double> ae(batch_size * h);
  for (int i = 0; i < batch_size * h; i++) {
    ae[i] = abs(test[i] - forecast_h[i]);
  }
  std::sort(ae.begin(), ae.end());
  double mae;
  if (h % 2 == 0) {
    mae = (ae[h / 2 - 1] + ae[h / 2]) / 2;
  } else {
    mae = ae[(int)h / 2];
  }
  std::cout << "MAE: " << mae << std::endl;
}

INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestAF,
                        ::testing::ValuesIn(inputsf1));
INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestMF,
                        ::testing::ValuesIn(inputsf2));
INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestMD,
                        ::testing::ValuesIn(inputsd2));

}  // namespace ML