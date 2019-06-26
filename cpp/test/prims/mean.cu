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
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
#include "random/rng.h"
#include "stats/mean.h"
#include "test_utils.h"

namespace MLCommon {
namespace Stats {

template <typename T>
struct MeanInputs {
  T tolerance, mean;
  int rows, cols;
  bool sample, rowMajor;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const MeanInputs<T> &dims) {
  return os;
}

template <typename T>
class MeanTest : public ::testing::TestWithParam<MeanInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MeanInputs<T>>::GetParam();
    Random::Rng r(params.seed);

    int rows = params.rows, cols = params.cols;
    int len = rows * cols;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocate(data, len);
    allocate(mean_act, cols);
    r.normal(data, len, params.mean, (T)1.0, stream);

    meanSGtest(data, stream);
  }

  void meanSGtest(T *data, cudaStream_t stream) {
    int rows = params.rows, cols = params.cols;

    mean(mean_act, data, cols, rows, params.sample, params.rowMajor, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(mean_act));
  }

 protected:
  MeanInputs<T> params;
  T *data, *mean_act;
};

// Note: For 1024 samples, 256 experiments, a mean of 1.0 with stddev=1.0, the
// measured mean (of a normal distribution) will fall outside of an epsilon of
// 0.15 only 4/10000 times. (epsilon of 0.1 will fail 30/100 times)
const std::vector<MeanInputs<float>> inputsf = {
  {0.15f, 1.f, 1024, 32, true, false, 1234ULL},
  {0.15f, 1.f, 1024, 64, true, false, 1234ULL},
  {0.15f, 1.f, 1024, 128, true, false, 1234ULL},
  {0.15f, 1.f, 1024, 256, true, false, 1234ULL},
  {0.15f, -1.f, 1024, 32, false, false, 1234ULL},
  {0.15f, -1.f, 1024, 64, false, false, 1234ULL},
  {0.15f, -1.f, 1024, 128, false, false, 1234ULL},
  {0.15f, -1.f, 1024, 256, false, false, 1234ULL},
  {0.15f, 1.f, 1024, 32, true, true, 1234ULL},
  {0.15f, 1.f, 1024, 64, true, true, 1234ULL},
  {0.15f, 1.f, 1024, 128, true, true, 1234ULL},
  {0.15f, 1.f, 1024, 256, true, true, 1234ULL},
  {0.15f, -1.f, 1024, 32, false, true, 1234ULL},
  {0.15f, -1.f, 1024, 64, false, true, 1234ULL},
  {0.15f, -1.f, 1024, 128, false, true, 1234ULL},
  {0.15f, -1.f, 1024, 256, false, true, 1234ULL}};

const std::vector<MeanInputs<double>> inputsd = {
  {0.15, 1.0, 1024, 32, true, false, 1234ULL},
  {0.15, 1.0, 1024, 64, true, false, 1234ULL},
  {0.15, 1.0, 1024, 128, true, false, 1234ULL},
  {0.15, 1.0, 1024, 256, true, false, 1234ULL},
  {0.15, -1.0, 1024, 32, false, false, 1234ULL},
  {0.15, -1.0, 1024, 64, false, false, 1234ULL},
  {0.15, -1.0, 1024, 128, false, false, 1234ULL},
  {0.15, -1.0, 1024, 256, false, false, 1234ULL},
  {0.15, 1.0, 1024, 32, true, true, 1234ULL},
  {0.15, 1.0, 1024, 64, true, true, 1234ULL},
  {0.15, 1.0, 1024, 128, true, true, 1234ULL},
  {0.15, 1.0, 1024, 256, true, true, 1234ULL},
  {0.15, -1.0, 1024, 32, false, true, 1234ULL},
  {0.15, -1.0, 1024, 64, false, true, 1234ULL},
  {0.15, -1.0, 1024, 128, false, true, 1234ULL},
  {0.15, -1.0, 1024, 256, false, true, 1234ULL}};

typedef MeanTest<float> MeanTestF;
TEST_P(MeanTestF, Result) {
  ASSERT_TRUE(devArrMatch(params.mean, mean_act, params.cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef MeanTest<double> MeanTestD;
TEST_P(MeanTestD, Result) {
  ASSERT_TRUE(devArrMatch(params.mean, mean_act, params.cols,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MeanTests, MeanTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MeanTests, MeanTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Stats
}  // end namespace MLCommon
