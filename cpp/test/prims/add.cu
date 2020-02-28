/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

template <typename InT, typename OutT = InT>
class AddTest : public ::testing::TestWithParam<AddInputs<InT, OutT>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<AddInputs<InT, OutT>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.len;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(in1, len);
    allocate(in2, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in1, len, InT(-1.0), InT(1.0), stream);
    r.uniform(in2, len, InT(-1.0), InT(1.0), stream);
    naiveAddElem<InT, OutT>(out_ref, in1, in2, len);
    add<InT, OutT>(out, in1, in2, len, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(in1));
    CUDA_CHECK(cudaFree(in2));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void compare() {
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<OutT>(params.tolerance)));
  }

 protected:
  AddInputs<InT, OutT> params;
  InT *in1, *in2;
  OutT *out_ref, *out;
  cudaStream_t stream;
};

const std::vector<AddInputs<float>> inputsf = {
  {0.000001f, 1024 * 1024, 1234ULL},
  {0.000001f, 1024 * 1024 + 2, 1234ULL},
  {0.000001f, 1024 * 1024 + 1, 1234ULL},
};
typedef AddTest<float> AddTestF;
TEST_P(AddTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(AddTests, AddTestF, ::testing::ValuesIn(inputsf));

const std::vector<AddInputs<double>> inputsd = {
  {0.00000001, 1024 * 1024, 1234ULL},
  {0.00000001, 1024 * 1024 + 2, 1234ULL},
  {0.00000001, 1024 * 1024 + 1, 1234ULL},
};
typedef AddTest<double> AddTestD;
TEST_P(AddTestD, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(AddTests, AddTestD, ::testing::ValuesIn(inputsd));

const std::vector<AddInputs<float, double>> inputsfd = {
  {0.00000001, 1024 * 1024, 1234ULL},
  {0.00000001, 1024 * 1024 + 2, 1234ULL},
  {0.00000001, 1024 * 1024 + 1, 1234ULL},
};
typedef AddTest<float, double> AddTestFD;
TEST_P(AddTestFD, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(AddTests, AddTestFD, ::testing::ValuesIn(inputsfd));

}  // end namespace LinAlg
}  // end namespace MLCommon
