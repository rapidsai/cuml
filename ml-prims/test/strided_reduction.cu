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
#include "linalg/strided_reduction.h"
#include "reduce.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
struct stridedReductionInputs {
    T tolerance;
    int rows, cols;
    unsigned long long int seed;
};

template <typename T>
void stridedReductionLaunch(T *dots, const T *data, int cols, int rows) {
  stridedReduction(dots, data, cols, rows, (T)0, false, 0,
                   [] __device__(T in, int i) { return in * in; });
}


template <typename T>
class stridedReductionTest : public ::testing::TestWithParam<stridedReductionInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<stridedReductionInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows*cols;

    allocate(data, len);
    allocate(dots_exp, cols); //expected dot products (from test)
    allocate(dots_act, cols); //actual dot products (from prim)
    r.uniform(data, len, T(-1.0), T(1.0)); //initialize matrix to random

    unaryAndGemv(dots_exp, data, cols, rows);
    stridedReductionLaunch(dots_act, data, cols, rows);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(dots_exp));
    CUDA_CHECK(cudaFree(dots_act));
  }

protected:
  stridedReductionInputs<T> params;
  T *data, *dots_exp, *dots_act;
};


const std::vector<stridedReductionInputs<float>> inputsf = {
  {0.00001f, 1024,  32, 1234ULL},
  {0.00001f, 1024,  64, 1234ULL},
  {0.00001f, 1024, 128, 1234ULL},
  {0.00001f, 1024, 256, 1234ULL}};

const std::vector<stridedReductionInputs<double>> inputsd = {
  {0.000000001, 1024,  32, 1234ULL},
  {0.000000001, 1024,  64, 1234ULL},
  {0.000000001, 1024, 128, 1234ULL},
  {0.000000001, 1024, 256, 1234ULL}};

typedef stridedReductionTest<float> stridedReductionTestF;
TEST_P(stridedReductionTestF, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef stridedReductionTest<double> stridedReductionTestD;
TEST_P(stridedReductionTestD, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.cols,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(stridedReductionTests, stridedReductionTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(stridedReductionTests, stridedReductionTestD, ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
