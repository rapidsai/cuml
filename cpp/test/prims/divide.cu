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
#include "test_utils.h"
#include "linalg/divide.h"
#include "random/rng.h"
#include "unary_op.h"

namespace MLCommon {
namespace LinAlg {

template <typename Type>
__global__ void naiveDivideKernel(Type *out, const Type *in, Type scalar,
                                  int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = in[idx] / scalar;
  }
}

template <typename Type>
void naiveDivide(Type *out, const Type *in, Type scalar, int len, cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  naiveDivideKernel<Type><<<nblks, TPB, 0, stream>>>(out, in, scalar, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template<typename T>
class DivideTest : public ::testing::TestWithParam<UnaryOpInputs<T>> {
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
    naiveDivide(out_ref, in, params.scalar, len, stream);
    divideScalar(out, in, params.scalar, len, stream);
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
typedef DivideTest<float> DivideTestF;
TEST_P(DivideTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DivideTests, DivideTestF,
                    ::testing::ValuesIn(inputsf));

typedef DivideTest<double> DivideTestD;
const std::vector<UnaryOpInputs<double>> inputsd = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
TEST_P(DivideTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DivideTests, DivideTestD,
                    ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
