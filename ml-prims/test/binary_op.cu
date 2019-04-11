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
template <typename T, typename IdxType>
void binaryOpLaunch(T *out, const T *in1, const T *in2, IdxType len, cudaStream_t stream) {
  binaryOp(out, in1, in2, len, [] __device__(T a, T b) { return a + b; }, stream);
}

template <typename T, typename IdxType>
class BinaryOpTest : public ::testing::TestWithParam<BinaryOpInputs<T, IdxType>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<BinaryOpInputs<T, IdxType>>::GetParam();
    Random::Rng r(params.seed);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    IdxType len = params.len;
    allocate(in1, len);
    allocate(in2, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in1, len, T(-1.0), T(1.0), stream);
    r.uniform(in2, len, T(-1.0), T(1.0), stream);
    naiveAdd(out_ref, in1, in2, len);
    binaryOpLaunch(out, in1, in2, len, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in1));
    CUDA_CHECK(cudaFree(in2));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

protected:
  BinaryOpInputs<T, IdxType> params;
  T *in1, *in2, *out_ref, *out;
};

const std::vector<BinaryOpInputs<float, int>> inputsf_i32 = {
  {0.000001f, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<float, int> BinaryOpTestF_i32;
TEST_P(BinaryOpTestF_i32, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BinaryOpTests, BinaryOpTestF_i32,
                        ::testing::ValuesIn(inputsf_i32));

const std::vector<BinaryOpInputs<float, size_t>> inputsf_i64 = {
  {0.000001f, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<float, size_t> BinaryOpTestF_i64;
TEST_P(BinaryOpTestF_i64, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BinaryOpTests, BinaryOpTestF_i64,
                        ::testing::ValuesIn(inputsf_i64));

const std::vector<BinaryOpInputs<double, int>> inputsd_i32 = {
  {0.00000001, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<double, int> BinaryOpTestD_i32;
TEST_P(BinaryOpTestD_i32, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BinaryOpTests, BinaryOpTestD_i32,
                        ::testing::ValuesIn(inputsd_i32));

const std::vector<BinaryOpInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<double, size_t> BinaryOpTestD_i64;
TEST_P(BinaryOpTestD_i64, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BinaryOpTests, BinaryOpTestD_i64,
                        ::testing::ValuesIn(inputsd_i64));

} // end namespace LinAlg
} // end namespace MLCommon
