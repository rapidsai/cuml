/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include "cuda_utils.h"
#include "reduce.h"
#include "linalg/reduce.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

template <typename T>
struct ReduceInputs {
  T tolerance;
  int rows, cols;
  bool rowMajor, alongRows;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const ReduceInputs<T> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void reduceLaunch(T *dots, const T *data, int cols, int rows, bool rowMajor,
                  bool alongRows, bool inplace, cudaStream_t stream) {
  reduce(dots, data, cols, rows, (T)0, rowMajor, alongRows, stream, inplace,
         [] __device__(T in, int i) { return in * in; });
}

template <typename T>
class ReduceTest : public ::testing::TestWithParam<ReduceInputs<T>> {
protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    params = ::testing::TestWithParam<ReduceInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    outlen = params.alongRows? rows : cols;
    allocate(data, len);
    allocate(dots_exp, outlen);
    allocate(dots_act, outlen);
    r.uniform(data, len, T(-1.0), T(1.0), stream);
    naiveReduction(dots_exp, data, cols, rows, params.rowMajor, params.alongRows,
                   stream);

    // Perform reduction with default inplace = false first
    reduceLaunch(dots_act, data, cols, rows, params.rowMajor, params.alongRows,
                 false, stream);
    // Add to result with inplace = true next, which shouldn't affect
    // in the case of coalescedReduction!
    if(!(params.rowMajor ^ params.alongRows)) {
      reduceLaunch(dots_act, data, cols, rows, params.rowMajor, params.alongRows,
                   true, stream);
    }
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(dots_exp));
    CUDA_CHECK(cudaFree(dots_act));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

protected:
  ReduceInputs<T> params;
  T *data, *dots_exp, *dots_act;
  int outlen;
  cudaStream_t stream;
};

const std::vector<ReduceInputs<float>> inputsf = {
  {0.000002f, 1024,  32, true, true, 1234ULL},
  {0.000002f, 1024,  64, true, true, 1234ULL},
  {0.000002f, 1024, 128, true, true, 1234ULL},
  {0.000002f, 1024, 256, true, true, 1234ULL},
  {0.000002f, 1024,  32, true, false, 1234ULL},
  {0.000002f, 1024,  64, true, false, 1234ULL},
  {0.000002f, 1024, 128, true, false, 1234ULL},
  {0.000002f, 1024, 256, true, false, 1234ULL},
  {0.000002f, 1024,  32, false, true, 1234ULL},
  {0.000002f, 1024,  64, false, true, 1234ULL},
  {0.000002f, 1024, 128, false, true, 1234ULL},
  {0.000002f, 1024, 256, false, true, 1234ULL},
  {0.000002f, 1024,  32, false, false, 1234ULL},
  {0.000002f, 1024,  64, false, false, 1234ULL},
  {0.000002f, 1024, 128, false, false, 1234ULL},
  {0.000002f, 1024, 256, false, false, 1234ULL}};

const std::vector<ReduceInputs<double>> inputsd = {
  {0.000000001, 1024,  32, true, true, 1234ULL},
  {0.000000001, 1024,  64, true, true, 1234ULL},
  {0.000000001, 1024, 128, true, true, 1234ULL},
  {0.000000001, 1024, 256, true, true, 1234ULL},
  {0.000000001, 1024,  32, true, false, 1234ULL},
  {0.000000001, 1024,  64, true, false, 1234ULL},
  {0.000000001, 1024, 128, true, false, 1234ULL},
  {0.000000001, 1024, 256, true, false, 1234ULL},
  {0.000000001, 1024,  32, false, true, 1234ULL},
  {0.000000001, 1024,  64, false, true, 1234ULL},
  {0.000000001, 1024, 128, false, true, 1234ULL},
  {0.000000001, 1024, 256, false, true, 1234ULL},
  {0.000000001, 1024,  32, false, false, 1234ULL},
  {0.000000001, 1024,  64, false, false, 1234ULL},
  {0.000000001, 1024, 128, false, false, 1234ULL},
  {0.000000001, 1024, 256, false, false, 1234ULL}};

typedef ReduceTest<float> ReduceTestF;
TEST_P(ReduceTestF, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, outlen,
                          CompareApprox<float>(params.tolerance)));
}

typedef ReduceTest<double> ReduceTestD;
TEST_P(ReduceTestD, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, outlen,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(ReduceTests, ReduceTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(ReduceTests, ReduceTestD, ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
