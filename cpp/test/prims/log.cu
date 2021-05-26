/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <raft/cudart_utils.h>
#include <functions/log.cuh>
#include <raft/cuda_utils.cuh>
#include "test_utils.h"

namespace MLCommon {
namespace Functions {

template <typename T>
struct LogInputs {
  T tolerance;
  int len;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const LogInputs<T>& dims) {
  return os;
}

template <typename T>
class LogTest : public ::testing::TestWithParam<LogInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<LogInputs<T>>::GetParam();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int len = params.len;

    raft::allocate(data, len);
    T data_h[params.len] = {2.1, 4.5, 0.34, 10.0};
    raft::update_device(data, data_h, len, stream);

    raft::allocate(result, len);
    raft::allocate(result_ref, len);
    T result_ref_h[params.len] = {0.74193734, 1.5040774, -1.07880966,
                                  2.30258509};
    raft::update_device(result_ref, result_ref_h, len, stream);

    f_log(result, data, T(1), len, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(result_ref));
  }

 protected:
  LogInputs<T> params;
  T *data, *result, *result_ref;
};

const std::vector<LogInputs<float>> inputsf2 = {{0.001f, 4}};

const std::vector<LogInputs<double>> inputsd2 = {{0.001, 4}};

typedef LogTest<float> LogTestValF;
TEST_P(LogTestValF, Result) {
  ASSERT_TRUE(devArrMatch(result_ref, result, params.len,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef LogTest<double> LogTestValD;
TEST_P(LogTestValD, Result) {
  ASSERT_TRUE(devArrMatch(result_ref, result, params.len,
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(LogTests, LogTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(LogTests, LogTestValD, ::testing::ValuesIn(inputsd2));

}  // end namespace Functions
}  // end namespace MLCommon
