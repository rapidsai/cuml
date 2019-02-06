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
#include "add.h"
#include "linalg/add.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {


template <typename T>
class AddTest : public ::testing::TestWithParam<AddInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<AddInputs<T>>::GetParam();
    Random::Rng<T> r(params.seed);
    int len = params.len;
    allocate(in1, len);
    allocate(in2, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in1, len, T(-1.0), T(1.0));
    r.uniform(in2, len, T(-1.0), T(1.0));
    naiveAddElem(out_ref, in1, in2, len);
    add(out, in1, in2, len);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in1));
    CUDA_CHECK(cudaFree(in2));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

protected:
  AddInputs<T> params;
  T *in1, *in2, *out_ref, *out;
};


const std::vector<AddInputs<float>> inputsf2 = {
  {0.000001f, 1024 * 1024, 1234ULL}};
typedef AddTest<float> AddTestF;
TEST_P(AddTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(AddTests, AddTestF, ::testing::ValuesIn(inputsf2));


const std::vector<AddInputs<double>> inputsd2 = {
  {0.00000001, 1024 * 1024, 1234ULL}};
typedef AddTest<double> AddTestD;
TEST_P(AddTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(AddTests, AddTestD, ::testing::ValuesIn(inputsd2));

} // end namespace LinAlg
} // end namespace MLCommon
