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
#include "matrix/matrix.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace Matrix {

template <typename T>
struct MatrixInputs {
  T tolerance;
  int n_row;
  int n_col;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const MatrixInputs<T> &dims) {
  return os;
}

template <typename T>
class MatrixTest : public ::testing::TestWithParam<MatrixInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MatrixInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.n_row * params.n_col;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(in1, len);
    allocate(in2, len);
    allocate(in1_revr, len);
    r.uniform(in1, len, T(-1.0), T(1.0), stream);

    copy(in1, in2, params.n_row, params.n_col, stream);
    // copy(in1, in1_revr, params.n_row, params.n_col);
    // colReverse(in1_revr, params.n_row, params.n_col);

    T *outTrunc;
    allocate(outTrunc, 6);
    truncZeroOrigin(in1, params.n_row, outTrunc, 3, 2, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in1));
    CUDA_CHECK(cudaFree(in2));
    // CUDA_CHECK(cudaFree(in1_revr));
  }

 protected:
  MatrixInputs<T> params;
  T *in1, *in2, *in1_revr;
};

const std::vector<MatrixInputs<float>> inputsf2 = {{0.000001f, 4, 4, 1234ULL}};

const std::vector<MatrixInputs<double>> inputsd2 = {
  {0.00000001, 4, 4, 1234ULL}};

typedef MatrixTest<float> MatrixTestF;
TEST_P(MatrixTestF, Result) {
  ASSERT_TRUE(devArrMatch(in1, in2, params.n_row * params.n_col,
                          CompareApprox<float>(params.tolerance)));
}

typedef MatrixTest<double> MatrixTestD;
TEST_P(MatrixTestD, Result) {
  ASSERT_TRUE(devArrMatch(in1, in2, params.n_row * params.n_col,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MatrixTests, MatrixTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(MatrixTests, MatrixTestD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace Matrix
}  // end namespace MLCommon
