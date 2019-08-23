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
#include "linalg/power.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename Type>
__global__ void naivePowerElemKernel(Type *out, const Type *in1,
                                     const Type *in2, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = myPow(in1[idx], in2[idx]);
  }
}

template <typename Type>
void naivePowerElem(Type *out, const Type *in1, const Type *in2, int len,
                    cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  naivePowerElemKernel<Type><<<nblks, TPB, 0, stream>>>(out, in1, in2, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
__global__ void naivePowerScalarKernel(Type *out, const Type *in1,
                                       const Type in2, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = myPow(in1[idx], in2);
  }
}

template <typename Type>
void naivePowerScalar(Type *out, const Type *in1, const Type in2, int len,
                      cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  naivePowerScalarKernel<Type><<<nblks, TPB, 0, stream>>>(out, in1, in2, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
struct PowerInputs {
  T tolerance;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const PowerInputs<T> &dims) {
  return os;
}

template <typename T>
class PowerTest : public ::testing::TestWithParam<PowerInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<PowerInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.len;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(in1, len);
    allocate(in2, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in1, len, T(1.0), T(2.0), stream);
    r.uniform(in2, len, T(1.0), T(2.0), stream);

    naivePowerElem(out_ref, in1, in2, len, stream);
    naivePowerScalar(out_ref, out_ref, T(2), len, stream);

    power(out, in1, in2, len, stream);
    powerScalar(out, out, T(2), len, stream);
    power(in1, in1, in2, len, stream);
    powerScalar(in1, in1, T(2), len, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in1));
    CUDA_CHECK(cudaFree(in2));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

 protected:
  PowerInputs<T> params;
  T *in1, *in2, *out_ref, *out;
  int device_count = 0;
};

const std::vector<PowerInputs<float>> inputsf2 = {
  {0.000001f, 1024 * 1024, 1234ULL}};

const std::vector<PowerInputs<double>> inputsd2 = {
  {0.00000001, 1024 * 1024, 1234ULL}};

typedef PowerTest<float> PowerTestF;
TEST_P(PowerTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ref, in1, params.len,
                          CompareApprox<float>(params.tolerance)));
}

typedef PowerTest<double> PowerTestD;
TEST_P(PowerTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(devArrMatch(out_ref, in1, params.len,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(PowerTests, PowerTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PowerTests, PowerTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace LinAlg
}  // end namespace MLCommon
