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
#include "matrix_vector_op.h"
#include "random/rng.h"
#include "stats/mean.h"
#include "stats/mean_center.h"
#include "test_utils.h"


namespace MLCommon {
namespace Stats {

template <typename T>
struct MeanCenterInputs {
  T tolerance, mean;
  int rows, cols;
  bool sample, rowMajor, bcastAlongRows;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os,
                           const MeanCenterInputs<T> &dims) {
  return os;
}

template <typename T>
class MeanCenterTest : public ::testing::TestWithParam<MeanCenterInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MeanCenterInputs<T>>::GetParam();
    Random::Rng<T> r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    allocate(out, len);
    allocate(out_ref, len);
    allocate(data, len);
    allocate(meanVec, cols);
    r.normal(data, len, params.mean, (T)1.0);
    mean(meanVec, data, cols, rows, params.sample, params.rowMajor);
    meanCenter(out, data, meanVec, cols, rows, params.rowMajor,
               params.bcastAlongRows);
    LinAlg::naiveMatVec(out_ref, data, meanVec, cols, rows, params.rowMajor,
                        params.bcastAlongRows, (T)-1.0);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(meanVec));
  }

protected:
  MeanCenterInputs<T> params;
  T *data, *meanVec, *out, *out_ref;
};


const std::vector<MeanCenterInputs<float>> inputsf = {
  {0.05f, 1.f, 1024, 32, true, false, true, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, false, true, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, false, true, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, false, true, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, false, true, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, false, true, 1234ULL},
  {0.05f, 1.f, 1024, 32, true, true, true, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, true, true, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, true, true, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, true, true, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, true, true, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, true, true, 1234ULL},
  {0.05f, 1.f, 1024, 32, true, false, false, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, false, false, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, false, false, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, false, false, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, false, false, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, false, false, 1234ULL},
  {0.05f, 1.f, 1024, 32, true, true, false, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, true, false, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, true, false, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, true, false, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, true, false, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, true, false, 1234ULL}};
typedef MeanCenterTest<float> MeanCenterTestF;
TEST_P(MeanCenterTestF, Result) {
  ASSERT_TRUE(devArrMatch(out, out_ref, params.cols,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MeanCenterTests, MeanCenterTestF,
                        ::testing::ValuesIn(inputsf));


const std::vector<MeanCenterInputs<double>> inputsd = {
  {0.05, 1.0, 1024, 32, true, false, true, 1234ULL},
  {0.05, 1.0, 1024, 64, true, false, true, 1234ULL},
  {0.05, 1.0, 1024, 128, true, false, true, 1234ULL},
  {0.05, -1.0, 1024, 32, false, false, true, 1234ULL},
  {0.05, -1.0, 1024, 64, false, false, true, 1234ULL},
  {0.05, -1.0, 1024, 128, false, false, true, 1234ULL},
  {0.05, 1.0, 1024, 32, true, true, true, 1234ULL},
  {0.05, 1.0, 1024, 64, true, true, true, 1234ULL},
  {0.05, 1.0, 1024, 128, true, true, true, 1234ULL},
  {0.05, -1.0, 1024, 32, false, true, true, 1234ULL},
  {0.05, -1.0, 1024, 64, false, true, true, 1234ULL},
  {0.05, -1.0, 1024, 128, false, true, true, 1234ULL},
  {0.05, 1.0, 1024, 32, true, false, false, 1234ULL},
  {0.05, 1.0, 1024, 64, true, false, false, 1234ULL},
  {0.05, 1.0, 1024, 128, true, false, false, 1234ULL},
  {0.05, -1.0, 1024, 32, false, false, false, 1234ULL},
  {0.05, -1.0, 1024, 64, false, false, false, 1234ULL},
  {0.05, -1.0, 1024, 128, false, false, false, 1234ULL},
  {0.05, 1.0, 1024, 32, true, true, false, 1234ULL},
  {0.05, 1.0, 1024, 64, true, true, false, 1234ULL},
  {0.05, 1.0, 1024, 128, true, true, false, 1234ULL},
  {0.05, -1.0, 1024, 32, false, true, false, 1234ULL},
  {0.05, -1.0, 1024, 64, false, true, false, 1234ULL},
  {0.05, -1.0, 1024, 128, false, true, false, 1234ULL}};
typedef MeanCenterTest<double> MeanCenterTestD;
TEST_P(MeanCenterTestD, Result) {
  ASSERT_TRUE(devArrMatch(out, out_ref, params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MeanCenterTests, MeanCenterTestD,
                        ::testing::ValuesIn(inputsd));

} // end namespace Stats
} // end namespace MLCommon
