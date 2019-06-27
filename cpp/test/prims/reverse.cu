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
#include "matrix/reverse.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace Matrix {

template <typename T>
struct ReverseInputs {
  T tolerance;
  int nrows, ncols;
  bool rowMajor, alongRows;
  unsigned long long seed;
};

template <typename T>
class ReverseTest : public ::testing::TestWithParam<ReverseInputs<T>> {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    params = ::testing::TestWithParam<ReverseInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.nrows * params.ncols;
    allocate(in, len);
    allocate(out, len);
    r.uniform(in, len, T(-1.0), T(1.0), stream);
    // applying reverse twice should yield the same output!
    // this will in turn also verify the inplace mode of reverse method
    reverse(out, in, params.nrows, params.ncols, params.rowMajor,
            params.alongRows, stream);
    reverse(out, out, params.nrows, params.ncols, params.rowMajor,
            params.alongRows, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  ReverseInputs<T> params;
  T *in, *out;
  cudaStream_t stream;
};

const std::vector<ReverseInputs<float>> inputsf = {
  {0.000001f, 32, 32, false, false, 1234ULL},
  {0.000001f, 32, 32, false, true, 1234ULL},
  {0.000001f, 32, 32, true, false, 1234ULL},
  {0.000001f, 32, 32, true, true, 1234ULL},

  {0.000001f, 41, 41, false, false, 1234ULL},
  {0.000001f, 41, 41, false, true, 1234ULL},
  {0.000001f, 41, 41, true, false, 1234ULL},
  {0.000001f, 41, 41, true, true, 1234ULL}};
typedef ReverseTest<float> ReverseTestF;
TEST_P(ReverseTestF, Result) {
  ASSERT_TRUE(devArrMatch(in, out, params.nrows, params.ncols,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReverseTests, ReverseTestF,
                        ::testing::ValuesIn(inputsf));

typedef ReverseTest<double> ReverseTestD;
const std::vector<ReverseInputs<double>> inputsd = {
  {0.000001, 32, 32, false, false, 1234ULL},
  {0.000001, 32, 32, false, true, 1234ULL},
  {0.000001, 32, 32, true, false, 1234ULL},
  {0.000001, 32, 32, true, true, 1234ULL},

  {0.000001, 41, 41, false, false, 1234ULL},
  {0.000001, 41, 41, false, true, 1234ULL},
  {0.000001, 41, 41, true, false, 1234ULL},
  {0.000001, 41, 41, true, true, 1234ULL}};
TEST_P(ReverseTestD, Result) {
  ASSERT_TRUE(devArrMatch(in, out, params.nrows, params.ncols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReverseTests, ReverseTestD,
                        ::testing::ValuesIn(inputsd));

}  // end namespace Matrix
}  // end namespace MLCommon
