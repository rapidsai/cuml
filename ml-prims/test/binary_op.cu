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
#include "binary_op.h"
#include "linalg/binary_op.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void binaryOpLaunch(T *out, const T *in1, const T *in2, int len) {
  binaryOp(out, in1, in2, len, [] __device__(T a, T b) { return a + b; });
}

template <typename T>
class BinaryOpTest : public ::testing::TestWithParam<BinaryOpInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<BinaryOpInputs<T>>::GetParam();
    Random::Rng<T> r(params.seed);
    int len = params.len;
    allocate(in1, len);
    allocate(in2, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in1, len, T(-1.0), T(1.0));
    r.uniform(in2, len, T(-1.0), T(1.0));
    naiveAdd(out_ref, in1, in2, len);
    binaryOpLaunch(out, in1, in2, len);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in1));
    CUDA_CHECK(cudaFree(in2));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

protected:
  BinaryOpInputs<T> params;
  T *in1, *in2, *out_ref, *out;
};

const std::vector<BinaryOpInputs<float>> inputsf = {
  {0.000001f, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<float> BinaryOpTestF;
TEST_P(BinaryOpTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BinaryOpTests, BinaryOpTestF,
                        ::testing::ValuesIn(inputsf));

const std::vector<BinaryOpInputs<double>> inputsd = {
  {0.00000001, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<double> BinaryOpTestD;
TEST_P(BinaryOpTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BinaryOpTests, BinaryOpTestD,
                        ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
