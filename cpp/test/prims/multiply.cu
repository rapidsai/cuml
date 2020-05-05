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

#include <common/cudart_utils.h>
#include <gtest/gtest.h>
#include "linalg/multiply.h"
#include "random/rng.h"
#include "test_utils.h"
#include "unary_op.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
class MultiplyTest : public ::testing::TestWithParam<UnaryOpInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<UnaryOpInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.len;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocate(in, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in, len, T(-1.0), T(1.0), stream);
    naiveScale(out_ref, in, params.scalar, len, stream);
    multiplyScalar(out, in, params.scalar, len, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

 protected:
  UnaryOpInputs<T> params;
  T *in, *out_ref, *out;
};

const std::vector<UnaryOpInputs<float>> inputsf = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
typedef MultiplyTest<float> MultiplyTestF;
TEST_P(MultiplyTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MultiplyTests, MultiplyTestF,
                        ::testing::ValuesIn(inputsf));

typedef MultiplyTest<double> MultiplyTestD;
const std::vector<UnaryOpInputs<double>> inputsd = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
TEST_P(MultiplyTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MultiplyTests, MultiplyTestD,
                        ::testing::ValuesIn(inputsd));

}  // end namespace LinAlg
}  // end namespace MLCommon
