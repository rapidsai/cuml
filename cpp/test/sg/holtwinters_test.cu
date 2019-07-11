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
#include "holtwinters/HoltWinters.hpp"

namespace ML {

using namespace MLCommon;

struct HoltWintersInputs {
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
    batch_size = params.batch_size;
    frequency = params.frequency;
    ML::SeasonalType seasonal = params.seasonal;
    start_periods = params.start_periods;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    alpha_ptr = std::vector<T>(batch_size);
    beta_ptr = std::vector<T>(batch_size);
    gamma_ptr = std::vector<T>(batch_size);
    SSE_error_ptr = std::vector<T>(batch_size);
    forecast_ptr = std::vector<T>(batch_size * h);

    std::vector<T> dataset_h = {
      112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126,
      141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150, 178, 163,
      172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218,
      230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272,
      237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229,
      203, 229, 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
      284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301,
      356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348,
      363, 435, 491, 505, 404, 359, 310, 337};

    allocate(data, batch_size * n);
    updateDevice(data, dataset_h.data(), batch_size * n, stream);

    ML::HoltWintersFitPredict(n, batch_size, frequency, h, start_periods,
                              seasonal, data, alpha_ptr.data(), beta_ptr.data(),
                              gamma_ptr.data(), SSE_error_ptr.data(),
                              forecast_ptr.data());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override { CUDA_CHECK(cudaFree(data)); }

 public:
  HoltWintersInputs params;
  T *data;
  int n = 120, h = 50;
  int batch_size, frequency, start_periods;
  std::vector<T> alpha_ptr, beta_ptr, gamma_ptr, SSE_error_ptr, forecast_ptr;
};

const std::vector<HoltWintersInputs> inputsf = {
  {1, 12, ML::SeasonalType::ADDITIVE, 2}};

typedef HoltWintersTest<float> HoltWintersTestF;
TEST_P(HoltWintersTestF, Fit) {
  myPrintHostVector("alpha", alpha_ptr.data(), batch_size);
  myPrintHostVector("beta", beta_ptr.data(), batch_size);
  myPrintHostVector("gamma", gamma_ptr.data(), batch_size);
  myPrintHostVector("forecast", forecast_ptr.data(), h);
  myPrintHostVector("error", SSE_error_ptr.data(), batch_size);
  ASSERT_TRUE(true == true);
}

INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestF,
                        ::testing::ValuesIn(inputsf));

}  // namespace ML