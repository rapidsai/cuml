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

template <typename T, typename IdxType>
struct MeanCenterInputs {
  T tolerance, mean;
  IdxType rows, cols;
  bool sample, rowMajor, bcastAlongRows;
  unsigned long long int seed;
};

template <typename T, typename IdxType>
::std::ostream &operator<<(::std::ostream &os,
                           const MeanCenterInputs<T, IdxType> &dims) {
  return os;
}

template <typename T, typename IdxType>
class MeanCenterTest
  : public ::testing::TestWithParam<MeanCenterInputs<T, IdxType>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MeanCenterInputs<T, IdxType>>::GetParam();
    Random::Rng r(params.seed);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto rows = params.rows, cols = params.cols;
    auto len = rows * cols;
    IdxType vecLen = params.bcastAlongRows ? cols : rows;

    allocate(out, len);
    allocate(out_ref, len);
    allocate(data, len);
    allocate(meanVec, vecLen);
    r.normal(data, len, params.mean, (T)1.0, stream);
    mean(meanVec, data, cols, rows, params.sample, params.rowMajor, stream);
    meanCenter(out, data, meanVec, cols, rows, params.rowMajor,
               params.bcastAlongRows, stream);
    LinAlg::naiveMatVec(out_ref, data, meanVec, cols, rows, params.rowMajor,
                        params.bcastAlongRows, (T)-1.0);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(meanVec));
  }

 protected:
  MeanCenterInputs<T, IdxType> params;
  T *data, *meanVec, *out, *out_ref;
};

const std::vector<MeanCenterInputs<float, int>> inputsf_i32 = {
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
typedef MeanCenterTest<float, int> MeanCenterTestF_i32;
TEST_P(MeanCenterTestF_i32, Result) {
  ASSERT_TRUE(devArrMatch(out, out_ref, params.cols,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MeanCenterTests, MeanCenterTestF_i32,
                        ::testing::ValuesIn(inputsf_i32));

const std::vector<MeanCenterInputs<float, size_t>> inputsf_i64 = {
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
typedef MeanCenterTest<float, size_t> MeanCenterTestF_i64;
TEST_P(MeanCenterTestF_i64, Result) {
  ASSERT_TRUE(devArrMatch(out, out_ref, params.cols,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MeanCenterTests, MeanCenterTestF_i64,
                        ::testing::ValuesIn(inputsf_i64));

const std::vector<MeanCenterInputs<double, int>> inputsd_i32 = {
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
typedef MeanCenterTest<double, int> MeanCenterTestD_i32;
TEST_P(MeanCenterTestD_i32, Result) {
  ASSERT_TRUE(devArrMatch(out, out_ref, params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MeanCenterTests, MeanCenterTestD_i32,
                        ::testing::ValuesIn(inputsd_i32));

const std::vector<MeanCenterInputs<double, size_t>> inputsd_i64 = {
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
typedef MeanCenterTest<double, size_t> MeanCenterTestD_i64;
TEST_P(MeanCenterTestD_i64, Result) {
  ASSERT_TRUE(devArrMatch(out, out_ref, params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MeanCenterTests, MeanCenterTestD_i64,
                        ::testing::ValuesIn(inputsd_i64));

}  // end namespace Stats
}  // end namespace MLCommon
