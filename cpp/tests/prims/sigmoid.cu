/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "test_utils.h"

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <functions/sigmoid.cuh>
#include <gtest/gtest.h>

namespace MLCommon {
namespace Functions {

template <typename T>
struct SigmoidInputs {
  T tolerance;
  int len;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SigmoidInputs<T>& dims)
{
  return os;
}

template <typename T>
class SigmoidTest : public ::testing::TestWithParam<SigmoidInputs<T>> {
 protected:
  SigmoidTest() : data(0, stream), result(0, stream), result_ref(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<SigmoidInputs<T>>::GetParam();

    int len = params.len;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    data.resize(len, stream);
    T data_h[params.len] = {2.1, -4.5, -0.34, 10.0};
    raft::update_device(data.data(), data_h, len, stream);

    result.resize(len, stream);
    result_ref.resize(len, stream);
    T result_ref_h[params.len] = {0.89090318, 0.01098694, 0.41580948, 0.9999546};
    raft::update_device(result_ref.data(), result_ref_h, len, stream);

    sigmoid(result.data(), data.data(), len, stream);
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream = 0;
  SigmoidInputs<T> params;
  rmm::device_uvector<T> data, result, result_ref;
};

const std::vector<SigmoidInputs<float>> inputsf2 = {{0.001f, 4}};

const std::vector<SigmoidInputs<double>> inputsd2 = {{0.001, 4}};

typedef SigmoidTest<float> SigmoidTestValF;
TEST_P(SigmoidTestValF, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(result_ref.data(),
                                    result.data(),
                                    params.len,
                                    MLCommon::CompareApproxAbs<float>(params.tolerance)));
}

typedef SigmoidTest<double> SigmoidTestValD;
TEST_P(SigmoidTestValD, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(result_ref.data(),
                                    result.data(),
                                    params.len,
                                    MLCommon::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(SigmoidTests, SigmoidTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SigmoidTests, SigmoidTestValD, ::testing::ValuesIn(inputsd2));

}  // end namespace Functions
}  // end namespace MLCommon
