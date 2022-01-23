/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "test_utils.h"
#include <gtest/gtest.h>
#include <linalg/sqrt.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng.hpp>

namespace MLCommon {
namespace LinAlg {

template <typename Type>
__global__ void naiveSqrtElemKernel(Type* out, const Type* in1, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = raft::mySqrt(in1[idx]); }
}

template <typename Type>
void naiveSqrtElem(Type* out, const Type* in1, int len)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  naiveSqrtElemKernel<Type><<<nblks, TPB>>>(out, in1, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
struct SqrtInputs {
  T tolerance;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SqrtInputs<T>& dims)
{
  return os;
}

template <typename T>
class SqrtTest : public ::testing::TestWithParam<SqrtInputs<T>> {
 protected:
  SqrtTest() : in1(0, stream), out_ref(0, stream), out(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<SqrtInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    int len = params.len;
    in1.resize(len, stream);
    out_ref.resize(len, stream);
    out.resize(len, stream);
    r.uniform(in1.data(), len, T(1.0), T(2.0), stream);

    naiveSqrtElem(out_ref.data(), in1.data(), len);

    sqrt(out.data(), in1.data(), len, stream);
    sqrt(in1.data(), in1.data(), len, stream);
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream = 0;
  SqrtInputs<T> params;
  rmm::device_uvector<T> in1, out_ref, out;
  int device_count = 0;
};

const std::vector<SqrtInputs<float>> inputsf2 = {{0.000001f, 1024 * 1024, 1234ULL}};

const std::vector<SqrtInputs<double>> inputsd2 = {{0.00000001, 1024 * 1024, 1234ULL}};

typedef SqrtTest<float> SqrtTestF;
TEST_P(SqrtTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), in1.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
}

typedef SqrtTest<double> SqrtTestD;
TEST_P(SqrtTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), in1.data(), params.len, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(SqrtTests, SqrtTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SqrtTests, SqrtTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace LinAlg
}  // end namespace MLCommon
