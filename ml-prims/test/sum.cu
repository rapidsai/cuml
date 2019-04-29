/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include "linalg/eltwise.h"
#include "random/rng.h"
#include "stats/sum.h"
#include "test_utils.h"

namespace MLCommon {
namespace Stats {

template <typename T>
struct SumInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const SumInputs<T> &dims) {
  return os;
}

template <typename T>
class SumTest : public ::testing::TestWithParam<SumInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<SumInputs<T>>::GetParam();
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(data, len);

    T data_h[len];
    for (int i = 0; i < len; i++) {
      data_h[i] = T(1);
    }

    updateDevice(data, data_h, len, stream);

    allocate(sum_act, cols);
    sum(sum_act, data, cols, rows, false, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(sum_act));
  }

protected:
  SumInputs<T> params;
  T *data, *sum_act;
};

const std::vector<SumInputs<float>> inputsf = {{0.05f, 1024, 32, 1234ULL},
                                               {0.05f, 1024, 256, 1234ULL}};

const std::vector<SumInputs<double>> inputsd = {{0.05, 1024, 32, 1234ULL},
                                                {0.05, 1024, 256, 1234ULL}};

typedef SumTest<float> SumTestF;
TEST_P(SumTestF, Result) {
  ASSERT_TRUE(devArrMatch(float(params.rows), sum_act, params.cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef SumTest<double> SumTestD;
TEST_P(SumTestD, Result) {
  ASSERT_TRUE(devArrMatch(double(params.rows), sum_act, params.cols,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(SumTests, SumTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(SumTests, SumTestD, ::testing::ValuesIn(inputsd));

} // end namespace Stats
} // end namespace MLCommon
