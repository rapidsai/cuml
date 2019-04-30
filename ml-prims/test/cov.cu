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
#include "random/rng.h"
#include "stats/cov.h"
#include "stats/mean.h"
#include "test_utils.h"


namespace MLCommon {
namespace Stats {

template <typename T>
struct CovInputs {
  T tolerance, mean, var;
  int rows, cols;
  bool sample, rowMajor, stable;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const CovInputs<T> &dims) {
  return os;
}

template <typename T>
class CovTest : public ::testing::TestWithParam<CovInputs<T>> {
protected:
  void SetUp() override {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    params = ::testing::TestWithParam<CovInputs<T>>::GetParam();
    params.tolerance *= 2;
    Random::Rng r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    T var = params.var;
    allocate(data, len);
    allocate(mean_act, cols);
    allocate(cov_act, cols * cols);
    r.normal(data, len, params.mean, var, stream);
    mean(mean_act, data, cols, rows, params.sample, params.rowMajor, stream);
    cov(cov_act, data, mean_act, cols, rows, params.sample, params.rowMajor,
        params.stable, handle, stream);

    T data_h[6] = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};
    T cov_cm_ref_h[4] = {4.3333, -2.8333, -2.8333, 2.333};

    allocate(data_cm, 6);
    allocate(cov_cm, 4);
    allocate(cov_cm_ref, 4);
    allocate(mean_cm, 2);

    updateDevice(data_cm, data_h, 6, stream);
    updateDevice(cov_cm_ref, cov_cm_ref_h, 4, stream);

    mean(mean_cm, data_cm, 2, 3, true, false, stream);
    cov(cov_cm, data_cm, mean_cm, 2, 3, true, false, true, handle, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(mean_act));
    CUDA_CHECK(cudaFree(cov_act));
    CUDA_CHECK(cudaFree(data_cm));
    CUDA_CHECK(cudaFree(cov_cm));
    CUDA_CHECK(cudaFree(cov_cm_ref));
    CUDA_CHECK(cudaFree(mean_cm));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

protected:
  CovInputs<T> params;
  T *data, *mean_act, *cov_act;
  cublasHandle_t handle;
  cudaStream_t stream;

  T *data_cm, *cov_cm, *cov_cm_ref, *mean_cm;
};

///@todo: add stable=false after it has been implemented
const std::vector<CovInputs<float>> inputsf = {
  {0.03f, 1.f, 2.f, 32 * 1024, 32, true, false, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 64, true, false, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 128, true, false, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 256, true, false, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 32, false, false, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 64, false, false, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 128, false, false, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 256, false, false, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 32, true, true, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 64, true, true, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 128, true, true, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 256, true, true, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 32, false, true, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 64, false, true, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 128, false, true, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 256, false, true, true, 1234ULL}};

const std::vector<CovInputs<double>> inputsd = {
  {0.03, 1.0, 2.0, 32 * 1024, 32, true, false, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 64, true, false, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 128, true, false, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 256, true, false, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 32, false, false, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 64, false, false, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 128, false, false, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 256, false, false, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 32, true, true, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 64, true, true, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 128, true, true, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 256, true, true, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 32, false, true, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 64, false, true, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 128, false, true, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 256, false, true, true, 1234ULL}};

typedef CovTest<float> CovTestF;
TEST_P(CovTestF, Result) {
  ASSERT_TRUE(diagonalMatch(params.var * params.var, cov_act, params.cols,
                            params.cols,
                            CompareApprox<float>(params.tolerance)));
}

typedef CovTest<double> CovTestD;
TEST_P(CovTestD, Result) {
  ASSERT_TRUE(diagonalMatch(params.var * params.var, cov_act, params.cols,
                            params.cols,
                            CompareApprox<double>(params.tolerance)));
}

typedef CovTest<float> CovTestSmallF;
TEST_P(CovTestSmallF, Result) {
  ASSERT_TRUE(devArrMatch(cov_cm_ref, cov_cm, 2, 2,
                          CompareApprox<float>(params.tolerance)));
}

typedef CovTest<double> CovTestSmallD;
TEST_P(CovTestSmallD, Result) {
  ASSERT_TRUE(devArrMatch(cov_cm_ref, cov_cm, 2, 2,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(CovTests, CovTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CovTests, CovTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_CASE_P(CovTests, CovTestSmallF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CovTests, CovTestSmallD, ::testing::ValuesIn(inputsd));

} // end namespace Stats
} // end namespace MLCommon
