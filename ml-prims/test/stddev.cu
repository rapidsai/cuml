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
#include "matrix/math.h"
#include "random/rng.h"
#include "stats/mean.h"
#include "stats/stddev.h"
#include "test_utils.h"


namespace MLCommon {
namespace Stats {

template <typename T>
struct StdDevInputs {
  T tolerance, mean, stddev;
  int rows, cols;
  bool sample, rowMajor;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const StdDevInputs<T> &dims) {
  return os;
}

template <typename T>
class StdDevTest : public ::testing::TestWithParam<StdDevInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<StdDevInputs<T>>::GetParam();
    Random::Rng<T> r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    allocate(data, len);
    allocate(mean_act, cols);
    allocate(stddev_act, cols);
    allocate(vars_act, cols);
    r.normal(data, len, params.mean, params.stddev);

    stdVarSGtest(data);
  }

  void stdVarSGtest(T *data) {
    int rows = params.rows, cols = params.cols;

    mean(mean_act, data, cols, rows, params.sample, params.rowMajor);

    stddev(stddev_act, data, mean_act, cols, rows, params.sample,
           params.rowMajor);

    vars(vars_act, data, mean_act, cols, rows, params.sample, params.rowMajor);

    Matrix::seqRoot(vars_act, T(1), cols);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(mean_act));
    CUDA_CHECK(cudaFree(stddev_act));
    CUDA_CHECK(cudaFree(vars_act));
  }

protected:
  StdDevInputs<T> params;
  T *data, *mean_act, *stddev_act, *vars_act;
};

const std::vector<StdDevInputs<float>> inputsf = {
  {0.1f, 1.f, 2.f, 1024, 32, true, false, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 64, true, false, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 128, true, false, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 256, true, false, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 32, false, false, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 64, false, false, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 128, false, false, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 256, false, false, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 32, true, true, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 64, true, true, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 128, true, true, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 256, true, true, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 32, false, true, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 64, false, true, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 128, false, true, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 256, false, true, 1234ULL}};

const std::vector<StdDevInputs<double>> inputsd = {
  {0.1, 1.0, 2.0, 1024, 32, true, false, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 64, true, false, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 128, true, false, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 256, true, false, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 32, false, false, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 64, false, false, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 128, false, false, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 256, false, false, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 32, true, true, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 64, true, true, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 128, true, true, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 256, true, true, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 32, false, true, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 64, false, true, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 128, false, true, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 256, false, true, 1234ULL}};

typedef StdDevTest<float> StdDevTestF;
TEST_P(StdDevTestF, Result) {
  ASSERT_TRUE(devArrMatch(params.stddev, stddev_act, params.cols,
                          CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(stddev_act, vars_act, params.cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef StdDevTest<double> StdDevTestD;
TEST_P(StdDevTestD, Result) {
  ASSERT_TRUE(devArrMatch(params.stddev, stddev_act, params.cols,
                          CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(stddev_act, vars_act, params.cols,
                          CompareApprox<double>(params.tolerance)));
}


INSTANTIATE_TEST_CASE_P(StdDevTests, StdDevTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(StdDevTests, StdDevTestD, ::testing::ValuesIn(inputsd));


} // end namespace Stats
} // end namespace MLCommon
