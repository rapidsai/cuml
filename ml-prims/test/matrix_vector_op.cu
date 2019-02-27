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
#include "matrix_vector_op.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

template <typename T>
struct MatVecOpInputs {
  T tolerance;
  int rows, cols;
  bool rowMajor, bcastAlongRows, useTwoVectors;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const MatVecOpInputs<T> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void matrixVectorOpLaunch(T *out, const T *in, const T *vec1, const T *vec2,
                          int D, int N, bool rowMajor, bool bcastAlongRows,
                          bool useTwoVectors) {
  if(useTwoVectors) {
    matrixVectorOp(out, in, vec1, vec2, D, N, rowMajor, bcastAlongRows,
                   [] __device__(T a, T b, T c) { return a + b + c; });
  } else {
    matrixVectorOp(out, in, vec1, D, N, rowMajor, bcastAlongRows,
                   [] __device__(T a, T b) { return a + b; });
  }
}

template <typename T>
class MatVecOpTest : public ::testing::TestWithParam<MatVecOpInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MatVecOpInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int N = params.rows, D = params.cols;
    int len = N * D;
    allocate(in, len);
    allocate(out_ref, len);
    allocate(out, len);
    int vecLen = params.bcastAlongRows ? D : N;
    allocate(vec1, vecLen);
    allocate(vec2, vecLen);
    r.uniform(in, len, (T)-1.0, (T)1.0);
    r.uniform(vec1, vecLen, (T)-1.0, (T)1.0);
    r.uniform(vec2, vecLen, (T)-1.0, (T)1.0);
    if(params.useTwoVectors) {
      naiveMatVec(out_ref, in, vec1, vec2, D, N, params.rowMajor,
                  params.bcastAlongRows, (T)1.0);
    } else {
      naiveMatVec(out_ref, in, vec1, D, N, params.rowMajor,
                  params.bcastAlongRows, (T)1.0);
    }
    matrixVectorOpLaunch(out, in, vec1, vec2, D, N, params.rowMajor,
                         params.bcastAlongRows, params.useTwoVectors);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(vec1));
    CUDA_CHECK(cudaFree(vec2));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(in));
  }

protected:
  MatVecOpInputs<T> params;
  T *in, *out, *out_ref, *vec1, *vec2;
};


const std::vector<MatVecOpInputs<float>> inputsf = {
  {0.00001f, 1024, 32, true, true, false, 1234ULL},
  {0.00001f, 1024, 64, true, true, false, 1234ULL},
  {0.00001f, 1024, 32, true, false, false, 1234ULL},
  {0.00001f, 1024, 64, true, false, false, 1234ULL},
  {0.00001f, 1024, 32, false, true, false, 1234ULL},
  {0.00001f, 1024, 64, false, true, false, 1234ULL},
  {0.00001f, 1024, 32, false, false, false, 1234ULL},
  {0.00001f, 1024, 64, false, false, false, 1234ULL},

  {0.00001f, 1024, 32, true, true, true, 1234ULL},
  {0.00001f, 1024, 64, true, true, true, 1234ULL},
  {0.00001f, 1024, 32, true, false, true, 1234ULL},
  {0.00001f, 1024, 64, true, false, true, 1234ULL},
  {0.00001f, 1024, 32, false, true, true, 1234ULL},
  {0.00001f, 1024, 64, false, true, true, 1234ULL},
  {0.00001f, 1024, 32, false, false, true, 1234ULL},
  {0.00001f, 1024, 64, false, false, true, 1234ULL}};
typedef MatVecOpTest<float> MatVecOpTestF;
TEST_P(MatVecOpTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.rows * params.cols,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MatVecOpTests, MatVecOpTestF,
                        ::testing::ValuesIn(inputsf));


const std::vector<MatVecOpInputs<double>> inputsd = {
  {0.0000001, 1024, 32, true, true, false, 1234ULL},
  {0.0000001, 1024, 64, true, true, false, 1234ULL},
  {0.0000001, 1024, 32, true, false, false, 1234ULL},
  {0.0000001, 1024, 64, true, false, false, 1234ULL},
  {0.0000001, 1024, 32, false, true, false, 1234ULL},
  {0.0000001, 1024, 64, false, true, false, 1234ULL},
  {0.0000001, 1024, 32, false, false, false, 1234ULL},
  {0.0000001, 1024, 64, false, false, false, 1234ULL},

  {0.0000001, 1024, 32, true, true, true, 1234ULL},
  {0.0000001, 1024, 64, true, true, true, 1234ULL},
  {0.0000001, 1024, 32, true, false, true, 1234ULL},
  {0.0000001, 1024, 64, true, false, true, 1234ULL},
  {0.0000001, 1024, 32, false, true, true, 1234ULL},
  {0.0000001, 1024, 64, false, true, true, 1234ULL},
  {0.0000001, 1024, 32, false, false, true, 1234ULL},
  {0.0000001, 1024, 64, false, false, true, 1234ULL}};
typedef MatVecOpTest<double> MatVecOpTestD;
TEST_P(MatVecOpTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.rows * params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MatVecOpTests, MatVecOpTestD,
                        ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
