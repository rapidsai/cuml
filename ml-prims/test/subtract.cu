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
#include "linalg/subtract.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename Type>
__global__ void naiveSubtractElemKernel(Type *out, const Type *in1,
                                        const Type *in2, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = in1[idx] - in2[idx];
  }
}

template <typename Type>
void naiveSubtractElem(Type *out, const Type *in1, const Type *in2, int len) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  naiveSubtractElemKernel<Type><<<nblks, TPB>>>(out, in1, in2, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
__global__ void naiveSubtractScalarKernel(Type *out, const Type *in1,
                                          const Type in2, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = in1[idx] - in2;
  }
}

template <typename Type>
void naiveSubtractScalar(Type *out, const Type *in1, const Type in2, int len) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  naiveSubtractScalarKernel<Type><<<nblks, TPB>>>(out, in1, in2, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
struct SubtractInputs {
  T tolerance;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const SubtractInputs<T> &dims) {
  return os;
}

template <typename T>
class SubtractTest : public ::testing::TestWithParam<SubtractInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<SubtractInputs<T>>::GetParam();
    Random::Rng<T> r(params.seed);
    int len = params.len;
    allocate(in1, len);
    allocate(in2, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in1, len, T(-1.0), T(1.0));
    r.uniform(in2, len, T(-1.0), T(1.0));

    naiveSubtractElem(out_ref, in1, in2, len);
    naiveSubtractScalar(out_ref, out_ref, T(1), len);

    subtract(out, in1, in2, len);
    subtractScalar(out, out, T(1), len);
    subtract(in1, in1, in2, len);
    subtractScalar(in1, in1, T(1), len);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in1));
    CUDA_CHECK(cudaFree(in2));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

protected:
  SubtractInputs<T> params;
  T *in1, *in2, *out_ref, *out;
};

const std::vector<SubtractInputs<float>> inputsf2 = {
  {0.000001f, 1024 * 1024, 1234ULL}};

const std::vector<SubtractInputs<double>> inputsd2 = {
  {0.00000001, 1024 * 1024, 1234ULL}};

typedef SubtractTest<float> SubtractTestF;
TEST_P(SubtractTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ref, in1, params.len,
                          CompareApprox<float>(params.tolerance)));
}

typedef SubtractTest<double> SubtractTestD;
TEST_P(SubtractTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ref, in1, params.len,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(SubtractTests, SubtractTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SubtractTests, SubtractTestD,
                        ::testing::ValuesIn(inputsd2));

} // end namespace LinAlg
} // end namespace MLCommon
